// Neighbours, Clark–Evans and parents on synthetic populations of 40 nuclei (as in the
// lecture's demo). Numbers are printed; pictures in out/relations/.
//   cargo run -p day16 -- relations                      (your code)
//   cargo run -p day16 --features solution -- relations  (the model solution)

use imgcourse::*;
use crate::relations::*;

const SIZE: usize = 256;

/// Nucleus centres and radii for a pattern: "clustered", "random" or "regular".
/// Nuclei never overlap (2 pixels apart at least) and stay 6 pixels inside the image.
fn place(pattern: &str, n: usize, seed: u64) -> Vec<(f32, f32, f32)> {
    let mut rng = Rng::new(seed);
    let clusters: Vec<(f32, f32)> = (0..5).map(|_| (40.0 + 176.0 * rng.uniform(), 40.0 + 176.0 * rng.uniform())).collect();
    let mut out: Vec<(f32, f32, f32)> = Vec::new();
    let mut k = 0;
    while out.len() < n {
        let r = 5.0 + 3.0 * rng.uniform();
        let (x, y) = match pattern {
            "clustered" => {
                let (cx, cy) = clusters[rng.below(5)];
                (cx + 12.0 * rng.normal(), cy + 12.0 * rng.normal())
            }
            "regular" => {
                // a 7 × 6 grid with a little jitter, in random order of filling
                k += 1;
                let i = (k - 1) % 42;
                (22.0 + 35.0 * (i % 7) as f32 + 4.0 * rng.normal(), 30.0 + 40.0 * (i / 7) as f32 + 4.0 * rng.normal())
            }
            _ => (SIZE as f32 * rng.uniform(), SIZE as f32 * rng.uniform()),
        };
        let inside = x > r + 6.0 && y > r + 6.0 && x < SIZE as f32 - r - 6.0 && y < SIZE as f32 - r - 6.0;
        let free = out.iter().all(|&(ox, oy, or)| ((x - ox).powi(2) + (y - oy).powi(2)).sqrt() > r + or + 2.0);
        if inside && free {
            out.push((x, y, r));
        }
    }
    out
}

/// Draw discs as a label image, 1..=n.
fn draw_discs(discs: &[(f32, f32, f32)]) -> Image<u32> {
    Image::from_fn(SIZE, SIZE, |x, y| {
        let mut l = 0;
        for (i, &(cx, cy, r)) in discs.iter().enumerate() {
            if (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2) <= r * r {
                l = i as u32 + 1;
            }
        }
        l
    })
}

/// Grow every object by `steps` pixels into the background (8-neighbours, one ring per step;
/// where two meet, the first label found wins): the "cell bodies" around the nuclei.
fn grow(labels: &Image<u32>, steps: usize) -> Image<u32> {
    let mut out = labels.clone();
    for _ in 0..steps {
        let prev = out.clone();
        out = prev.map_xy(|x, y, l| {
            if l != 0 {
                return l;
            }
            N8.iter().map(|&(dx, dy)| label_at(&prev, x as isize + dx, y as isize + dy)).find(|&m| m != 0).unwrap_or(0)
        });
    }
    out
}

fn centroids(labels: &Image<u32>, n: usize) -> Vec<[f32; 2]> {
    let mut s = vec![[0.0f32; 3]; n + 1];
    for (x, y) in labels.coords() {
        let l = labels[(x, y)] as usize;
        s[l] = [s[l][0] + x as f32, s[l][1] + y as f32, s[l][2] + 1.0];
    }
    (1..=n).map(|l| [s[l][0] / s[l][2], s[l][1] / s[l][2]]).collect()
}

fn mean_count(nb: &[std::collections::BTreeSet<u32>]) -> f32 {
    nb[1..].iter().map(|s| s.len()).sum::<usize>() as f32 / (nb.len() - 1) as f32
}

/// Draw a straight line from `a` to `b` (one dot per pixel step along the longer axis).
fn line(img: &mut Image<[f32; 3]>, a: [f32; 2], b: [f32; 2], colour: [f32; 3]) {
    let steps = (b[0] - a[0]).abs().max((b[1] - a[1]).abs()).ceil().max(1.0) as usize;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        img[((a[0] + t * (b[0] - a[0])).round() as usize, (a[1] + t * (b[1] - a[1])).round() as usize)] = colour;
    }
}

/// The regions in dim colours, the objects bright, and the neighbour graph in white.
fn graph_picture(regions: &Image<u32>, objects: &Image<u32>, nb: &[std::collections::BTreeSet<u32>], c: &[[f32; 2]]) -> Image<[f32; 3]> {
    let colours = io::label_colours(regions);
    let mut pic = colours.map_xy(|x, y, col| if objects[(x, y)] != 0 { col } else { col.map(|v| 0.35 * v) });
    for (a, set) in nb.iter().enumerate().skip(1) {
        for &b in set {
            if (b as usize) > a {
                line(&mut pic, c[a - 1], c[b as usize - 1], [1.0, 1.0, 1.0]);
            }
        }
    }
    pic
}

pub fn main() {
    let dir = "out/relations";
    let n = 40;
    let area = (SIZE * SIZE) as f32;
    println!("{n} nuclei in {SIZE} × {SIZE} px; mean number of neighbours by definition, and Clark–Evans R");
    println!("{:<10} {:>9} {:>9} {:>6}", "pattern", "adjacent", "expanded", "R");
    for pattern in ["clustered", "random", "regular"] {
        let nuclei = draw_discs(&place(pattern, n, 11));
        let cells = grow(&nuclei, 9);
        let expanded = expand_labels(&nuclei);
        let c = centroids(&nuclei, n);
        let (nb_adj, nb_exp) = (neighbours(&cells, false), neighbours(&expanded, false));
        println!("{:<10} {:>9.2} {:>9.2} {:>6.2}", pattern, mean_count(&nb_adj), mean_count(&nb_exp), clark_evans(&c, area));
        save_rgb(&graph_picture(&cells, &nuclei, &nb_adj, &c), format!("{dir}/{pattern}-adjacent.png"));
        save_rgb(&graph_picture(&expanded, &nuclei, &nb_exp, &c), format!("{dir}/{pattern}-expanded.png"));
    }

    // One cell of the random pattern in detail: closest two, angle, percent touching.
    let nuclei = draw_discs(&place("random", n, 11));
    let cells = grow(&nuclei, 9);
    let c = centroids(&nuclei, n);
    let close = closest_two(&c);
    let busiest = (1..=n as u32).max_by(|&a, &b| percent_touching(&cells, a).total_cmp(&percent_touching(&cells, b))).unwrap();
    let (f, d1, s, d2) = close[busiest as usize - 1];
    println!(
        "\ncell {busiest} (most touching): {:.0} % of its boundary touches other cells; closest {} at {d1:.1} px, second {} at {d2:.1} px, angle {:.0}°",
        percent_touching(&cells, busiest),
        f + 1,
        s + 1,
        angle_at(c[busiest as usize - 1], c[f], c[s])
    );

    // Parents and children: 50 small spots (radius 2), most inside a nucleus, some in the
    // cytoplasm. Parents: the nuclei, then the cells.
    let mut rng = Rng::new(4);
    let spots: Vec<(f32, f32, f32)> = (0..50)
        .map(|_| {
            let (cx, cy, r) = place("random", n, 11)[rng.below(n)];
            let (dist, angle) = (if rng.uniform() < 0.85 { r * rng.uniform() } else { r + 6.0 * rng.uniform() }, std::f32::consts::TAU * rng.uniform());
            (cx + dist * angle.cos(), cy + dist * angle.sin(), 2.0)
        })
        .collect();
    let spot_labels = draw_discs(&spots);
    for (name, parents) in [("nuclei", &nuclei), ("cells", &cells)] {
        let by_overlap = parent_by_overlap(&spot_labels, parents);
        let by_centroid = parent_by_centroid(&spot_labels, parents);
        let present: Vec<usize> = (1..by_overlap.len()).filter(|&c| spot_labels.pixels().contains(&(c as u32))).collect();
        let orphans = present.iter().filter(|&&c| by_overlap[c].0 == 0).count();
        let partial = present.iter().filter(|&&c| by_overlap[c].0 != 0 && by_overlap[c].1 < 1.0).count();
        let differ = present.iter().filter(|&&c| by_overlap[c].0 != by_centroid[c]).count();
        println!(
            "parents = {name:<6}: {} spots, {orphans} without a parent, {partial} only partly inside their parent, {differ} where overlap and centroid disagree",
            present.len()
        );
    }
    let pic = io::label_colours(&cells).map(|c| c.map(|v| 0.3 * v));
    let pic = pic.map_xy(|x, y, c| if spot_labels[(x, y)] != 0 { [1.0, 1.0, 1.0] } else if nuclei[(x, y)] != 0 { c.map(|v| 2.5 * v) } else { c });
    save_rgb(&pic, format!("{dir}/spots.png"));
    println!("wrote the pictures to {dir}/");
}
