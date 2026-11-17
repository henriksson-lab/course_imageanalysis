// Merge trees of a synthetic nuclei image (leaves = watershed basins of the gradient):
// altitude, dynamics, area and volume hierarchies, and greedy merging by difference of means.
// Each is cut at 20 regions and compared with the true nuclei. Pictures in out/merge-trees/.
//   cargo run -p merge-trees                      (your code)
//   cargo run -p merge-trees --features solution  (the model solution)

use imgcourse::*;
use merge_trees::*;
use std::time::Instant;

/// The gradient magnitude (Sobel) after a 3 × 3 binomial blur.
fn gradient(img: &Image<f32>) -> Image<f32> {
    let k = [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]];
    let blur = img.map_xy(|x, y, _| {
        let mut s = 0.0;
        for dy in 0..3 {
            for dx in 0..3 {
                s += k[dy][dx] / 16.0 * img.get_border(x as isize + dx as isize - 1, y as isize + dy as isize - 1, Border::Clamp);
            }
        }
        s
    });
    blur.map_xy(|x, y, _| {
        let v = |dx: isize, dy: isize| blur.get_border(x as isize + dx, y as isize + dy, Border::Clamp);
        let gx = v(1, -1) + 2.0 * v(1, 0) + v(1, 1) - v(-1, -1) - 2.0 * v(-1, 0) - v(-1, 1);
        let gy = v(-1, 1) + 2.0 * v(0, 1) + v(1, 1) - v(-1, -1) - 2.0 * v(0, -1) - v(1, -1);
        (gx * gx + gy * gy).sqrt()
    })
}

/// How many true nuclei have a region that overlaps them with IoU ≥ 0.5.
fn matched(labels: &[usize], truth: &Image<u32>) -> usize {
    let n = truth.pixels().iter().copied().max().unwrap_or(0);
    let regions = labels.iter().max().map_or(0, |m| m + 1);
    let mut region_size = vec![0usize; regions];
    for &l in labels {
        region_size[l] += 1;
    }
    (1..=n)
        .filter(|&t| {
            let mut counts = std::collections::HashMap::new();
            let mut size = 0;
            for (p, &v) in truth.pixels().iter().enumerate() {
                if v == t {
                    size += 1;
                    *counts.entry(labels[p]).or_insert(0usize) += 1;
                }
            }
            counts.iter().any(|(&l, &inter)| inter as f32 / (size + region_size[l] - inter) as f32 >= 0.5)
        })
        .count()
}

/// The regions in their mean grey level, with region boundaries in yellow.
fn picture(img: &Image<f32>, labels: &[usize]) -> Image<[f32; 3]> {
    let (w, h) = (img.width(), img.height());
    let regions = labels.iter().max().map_or(0, |m| m + 1);
    let (mut sum, mut count) = (vec![0.0f32; regions], vec![0.0f32; regions]);
    for (p, &l) in labels.iter().enumerate() {
        sum[l] += img.pixels()[p];
        count[l] += 1.0;
    }
    Image::from_fn(w, h, |x, y| {
        let l = labels[y * w + x];
        let edge = (x + 1 < w && labels[y * w + x + 1] != l) || (y + 1 < h && labels[(y + 1) * w + x] != l);
        if edge {
            [1.0, 0.9, 0.0]
        } else {
            let m = sum[l] / count[l];
            [m, m, m]
        }
    })
}

fn main() {
    let dir = "out/merge-trees";
    let (img, truth) = samples::nuclei(256, 18, 4);
    let (w, h) = (img.width(), img.height());
    let n = w * h;
    let f = gradient(&img);
    // the pixel graph: edges to the right and down at the pass max(f_p, f_q)
    let mut edges = Vec::new();
    for (x, y) in img.coords() {
        let p = y * w + x;
        if x + 1 < w {
            edges.push((p, p + 1, f[(x, y)].max(f[(x + 1, y)])));
        }
        if y + 1 < h {
            edges.push((p, p + w, f[(x, y)].max(f[(x, y + 1)])));
        }
    }
    save_gray(&img, format!("{dir}/nuclei.png"));

    // The leaves: watershed basins. Cutting the pixel-level dynamics tree at level 0 would give
    // one basin per regional minimum (thousands, because of the noise); we cut it at 150
    // regions instead, i.e. keep the 150 deepest basins.
    let start = Instant::now();
    let pixel_dynamics = attribute_tree(n, &edges, f.pixels(), &vec![1.0; n], Attribute::Dynamics);
    let basin = cut_regions(n, &pixel_dynamics, 150);
    let nb = basin.iter().max().map_or(0, |m| m + 1);
    // basin statistics and the passes between basins (the lowest pixel edge between them)
    let (mut minimum, mut area, mut sum) = (vec![f32::INFINITY; nb], vec![0.0f32; nb], vec![0.0f32; nb]);
    for p in 0..n {
        minimum[basin[p]] = minimum[basin[p]].min(f.pixels()[p]);
        area[basin[p]] += 1.0;
        sum[basin[p]] += img.pixels()[p];
    }
    let means: Vec<f32> = (0..nb).map(|b| sum[b] / area[b]).collect();
    let mut pass = std::collections::HashMap::new();
    for &(p, q, d) in &edges {
        let (a, b) = (basin[p].min(basin[q]), basin[p].max(basin[q]));
        if a != b {
            let e = pass.entry((a, b)).or_insert(f32::INFINITY);
            *e = e.min(d);
        }
    }
    let mut basin_edges: Vec<Edge> = pass.iter().map(|(&(a, b), &d)| (a, b, d)).collect();
    basin_edges.sort_by_key(|e| (e.0, e.1));
    let pairs: Vec<(usize, usize)> = basin_edges.iter().map(|&(a, b, _)| (a, b)).collect();
    println!("{w} × {h}, 18 nuclei: {nb} basins as leaves, {} basin pairs ({:.0} ms)", basin_edges.len(), 1e3 * start.elapsed().as_secs_f64());
    let to_pixels = |leaf_labels: &[usize]| -> Vec<usize> { basin.iter().map(|&b| leaf_labels[b]).collect() };

    let regions = 20;
    println!("each hierarchy cut at {regions} regions (nuclei matched with IoU ≥ 0.5):");
    let trees = [
        ("altitude", altitude_tree(nb, &basin_edges)),
        ("dynamics", attribute_tree(nb, &basin_edges, &minimum, &area, Attribute::Dynamics)),
        ("area", attribute_tree(nb, &basin_edges, &minimum, &area, Attribute::Area)),
        ("volume", attribute_tree(nb, &basin_edges, &minimum, &area, Attribute::Volume)),
        ("mean-diff", rag_merge(&means, &area, &pairs)),
    ];
    for (name, tree) in &trees {
        let labels = to_pixels(&cut_regions(nb, tree, regions));
        let best = (5..=60).map(|r| (matched(&to_pixels(&cut_regions(nb, tree, r)), &truth), r)).max_by_key(|&(m, r)| (m, std::cmp::Reverse(r))).unwrap();
        println!("  {name:9}: {:2} of 18; the best cut has {} at {} regions", matched(&labels, &truth), best.0, best.1);
        save_rgb(&picture(&img, &labels), format!("{dir}/{name}-{regions}.png"));
    }

    // The ultrametric contour map of the dynamics tree: every pixel on a basin boundary shows
    // the level at which that boundary disappears.
    let u = ucm(nb, &trees[1].1, &basin_edges);
    let value: std::collections::HashMap<(usize, usize), f32> = basin_edges.iter().zip(&u).map(|(&(a, b, _), &v)| ((a, b), v)).collect();
    let mut map = Image::new(w, h, 0.0f32);
    for &(p, q, _) in &edges {
        let (a, b) = (basin[p].min(basin[q]), basin[p].max(basin[q]));
        if a != b {
            for r in [p, q] {
                let (x, y) = (r % w, r / w);
                map[(x, y)] = map[(x, y)].max(value[&(a, b)]);
            }
        }
    }
    save_gray_stretched(&map, format!("{dir}/ucm-dynamics.png"));
    println!("\nwrote the pictures to {dir}/");
}
