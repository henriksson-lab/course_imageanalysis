// Finds keys in a synthetic scene with the generalised Hough transform: translation only, then
// a search over rotations and scales. Pictures in out/ght/.
//   cargo run -p day21 -- ght                      (your code)
//   cargo run -p day21 --features solution -- ght  (the model solution)

use crate::ght::sim::{edge_points, key, polygon_scene, show_points, transform};
use crate::ght::*;
use imgcourse::*;

/// The votes within one pixel of each cell (3 × 3), divided by the number of template edge
/// points. A long straight side puts several of its votes near the reference point, so this can
/// exceed 1; it is only used to find the candidates.
fn score_map(acc: &Image<f32>, n_template: usize) -> Image<f32> {
    acc.map_xy(|x, y, _| {
        let mut s = 0.0;
        for dy in -1..=1isize {
            for dx in -1..=1isize {
                if acc.contains(x as isize + dx, y as isize + dy) {
                    s += acc[((x as isize + dx) as usize, (y as isize + dy) as usize)];
                }
            }
        }
        s / n_template as f32
    })
}

/// A detection: x, y, the index of the pose, and the score.
type Detection = (usize, usize, usize, f32);

/// Search the poses: for each pixel the best score over all poses, and which pose gave it.
/// Then keep the local maxima (±6 px) above `threshold`: (x, y, pose index, score).
fn search(table: &RTable, edges: &[crate::ght::sim::EdgePoint], size: usize, poses: &[(f32, f32)], n_template: usize, threshold: f32) -> (Vec<Detection>, Image<f32>) {
    let mut best = Image::new(size, size, (0.0f32, 0usize));
    for (i, &(phi, s)) in poses.iter().enumerate() {
        let sc = score_map(&vote_pose(table, edges, size, size, phi, s), n_template);
        for (x, y) in sc.coords() {
            if sc[(x, y)] > best[(x, y)].0 {
                best[(x, y)] = (sc[(x, y)], i);
            }
        }
    }
    let score = best.map(|b| b.0);
    let mut found = Vec::new();
    for (x, y) in score.coords() {
        let v = score[(x, y)];
        if v < threshold {
            continue;
        }
        let mut is_max = true;
        for dy in -6..=6isize {
            for dx in -6..=6isize {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if (dx, dy) != (0, 0) && score.contains(nx, ny) {
                    let u = score[(nx as usize, ny as usize)];
                    if u > v || (u == v && (dy, dx) < (0, 0)) {
                        is_max = false;
                    }
                }
            }
        }
        if is_max {
            let pose = best[(x, y)].1;
            let honest = point_score(table, edges, poses[pose], x, y, n_template);
            if honest >= threshold {
                found.push((x, y, pose, honest));
            }
        }
    }
    found.sort_by(|a, b| b.3.total_cmp(&a.3));
    (found, score)
}

/// The score of a detection: the fraction of template edge points matched, i.e. the number of
/// scene edge points that cast at least one vote within one pixel of `(x, y)` for this pose,
/// divided by the number of template edge points.
fn point_score(table: &RTable, edges: &[crate::ght::sim::EdgePoint], pose: (f32, f32), x: usize, y: usize, n_template: usize) -> f32 {
    let (phi, s) = pose;
    let (c, sn) = (phi.cos(), phi.sin());
    let mut count = 0;
    for e in edges {
        let hit = table[bin_of(e.theta - phi, table.len())].iter().any(|v| {
            let (vx, vy) = (e.x + s * (c * v[0] - sn * v[1]), e.y + s * (sn * v[0] + c * v[1]));
            (vx.round() - x as f32).abs() <= 1.0 && (vy.round() - y as f32).abs() <= 1.0
        });
        if hit {
            count += 1;
        }
    }
    count as f32 / n_template as f32
}

pub fn main() {
    let dir = "out/ght";
    // The template: one key in a 64 × 64 frame, reference point at its centre (32, 32).
    let reference = [32.0, 32.0];
    let template = polygon_scene(64, 0.1, &key().into_iter().map(|p| (p, 0.8)).collect::<Vec<_>>(), 0.0, 1);
    let t_edges = edge_points(&template, 1.0, 0.25);
    let table = build_r_table(&t_edges, reference, 36);
    let n = t_edges.len();
    println!("template: {n} edge points, R-table with 36 bins ({} to {} vectors per bin)", table.iter().map(|b| b.len()).min().unwrap(), table.iter().map(|b| b.len()).max().unwrap());
    save_rgb(&show_points(&template, &t_edges), format!("{dir}/template.png"));

    // The scene: an upright key, one turned by 60°, one scaled by 1.2, one half hidden by a
    // dark disc, and some clutter blobs.
    let size = 192;
    let mut shapes: Vec<(Vec<[f32; 2]>, f32)> = Vec::new();
    let copies = [([10.0, 8.0], 0.0, 1.0), ([110.0, 20.0], 60f32.to_radians(), 1.0), ([20.0, 110.0], 0.0, 1.2), ([115.0, 115.0], 0.0, 1.0)];
    for &(shift, phi, s) in &copies {
        for p in transform(&key(), reference, phi, s, shift) {
            shapes.push((p, 0.8));
        }
    }
    let disc = |cx: f32, cy: f32, r: f32| (0..36).map(|i| {
        let a = i as f32 * std::f32::consts::TAU / 36.0;
        [cx + r * a.cos(), cy + r * a.sin()]
    }).collect::<Vec<_>>();
    shapes.push((disc(160.0, 147.0, 14.0), 0.1)); // hides the end of the fourth key
    for (cx, cy) in [(90.0, 90.0), (175.0, 60.0), (70.0, 170.0), (100.0, 10.0)] {
        shapes.push((disc(cx, cy, 7.0), 0.5));
    }
    let img = polygon_scene(size, 0.1, &shapes, 0.03, 2);
    let edges = edge_points(&img, 1.0, 0.25);
    println!("scene: {} edge points; true reference points (42, 40), (142, 52), (52, 142), (147, 147)", edges.len());

    let searches: [(&str, Vec<(f32, f32)>); 2] = [
        ("translation", vec![(0.0, 1.0)]),
        ("rotation-scale", (0..36).flat_map(|i| [0.8, 0.9, 1.0, 1.1, 1.2].map(|s| ((10 * i) as f32 * std::f32::consts::PI / 180.0, s))).collect()),
    ];
    for (name, poses) in &searches {
        let (all, score) = search(&table, &edges, size, poses, n, 0.3);
        let (found, weak): (Vec<_>, Vec<_>) = all.into_iter().partition(|f| f.3 >= 0.45);
        let lookups = edges.len() * poses.len();
        println!("{name}: {} poses, {lookups} table lookups, {} detections with score ≥ 0.45:", poses.len(), found.len());
        let mut out = show_points(&img, &edges);
        for &(x, y, i, v) in &found {
            let (phi, s) = poses[i];
            println!("   ({x:>3}, {y:>3}) angle {:>3.0}°, scale {s:.1}: {v:.2}", phi.to_degrees());
            // draw the template's edge points at the found pose
            for e in &t_edges {
                let (dx, dy) = (e.x - reference[0], e.y - reference[1]);
                let (c, sn) = (phi.cos(), phi.sin());
                let (px, py) = ((x as f32 + s * (c * dx - sn * dy)).round() as isize, (y as f32 + s * (sn * dx + c * dy)).round() as isize);
                if out.contains(px, py) {
                    out[(px as usize, py as usize)] = [0.0, 0.9, 1.0];
                }
            }
        }
        if let Some(w) = weak.first() {
            println!("   and {} weaker peaks between 0.30 and 0.45, the strongest {:.2} at ({}, {}), angle {:.0}°", weak.len(), w.3, w.0, w.1, poses[w.2].0.to_degrees());
        }
        save_gray_stretched(&score, format!("{dir}/accumulator-{name}.png"));
        save_rgb(&out, format!("{dir}/detections-{name}.png"));
    }
    println!("wrote the pictures to {dir}/");
}
