// Model solution for the coding exercises of "Objects instead of pixels".

use crate::sim::{dist, in_mask, Point, PIXEL};
use imgcourse::{Image, Rng};

/// C1a. For every A spot, the index of the nearest B spot and the distance to it
/// (slide "Distances between channels"). `b` is not empty.
pub fn nearest(a: &[Point], b: &[Point]) -> Vec<(usize, f32)> {
    let mut out = Vec::new();
    for &pa in a {
        let mut best = (0, f32::INFINITY);
        for (j, &pb) in b.iter().enumerate() {
            let d = dist(pa, pb);
            if d < best.1 {
                best = (j, d);
            }
        }
        out.push(best);
    }
    out
}

/// C1b. The share of A spots whose nearest B spot is at most `d_max` away.
pub fn fraction_within(a: &[Point], b: &[Point], d_max: f32) -> f32 {
    let close = nearest(a, b).iter().filter(|&&(_, d)| d <= d_max).count();
    close as f32 / a.len() as f32
}

/// C2. One-to-one pairing: the number of pairs (A spot, B spot) at most `d_max` apart when
/// every spot may be used only once. Take the closest pairs first (greedy): list all pairs
/// within `d_max`, sort them by distance, and accept a pair if neither spot is used yet.
pub fn one_to_one(a: &[Point], b: &[Point], d_max: f32) -> usize {
    let mut pairs = Vec::new();
    for (i, &pa) in a.iter().enumerate() {
        for (j, &pb) in b.iter().enumerate() {
            let d = dist(pa, pb);
            if d <= d_max {
                pairs.push((d, i, j));
            }
        }
    }
    pairs.sort_by(|p, q| p.0.total_cmp(&q.0));
    let (mut used_a, mut used_b) = (vec![false; a.len()], vec![false; b.len()]);
    let mut count = 0;
    for (_, i, j) in pairs {
        if !used_a[i] && !used_b[j] {
            used_a[i] = true;
            used_b[j] = true;
            count += 1;
        }
    }
    count
}

/// C3a. The chance that a random point has a B spot within `d`, for B spots placed at random
/// with density `rho` (spots per unit area): 1 − exp(−rho π d²) (slide "The chance level").
pub fn chance_level(rho: f32, d: f32) -> f32 {
    1.0 - (-rho * std::f32::consts::PI * d * d).exp()
}

/// C3b. The fraction corrected for chance: (f_obs − f_chance) / (1 − f_chance).
pub fn corrected(f_obs: f32, f_chance: f32) -> f32 {
    (f_obs - f_chance) / (1.0 - f_chance)
}

/// C4. The three overlap definitions for labelled objects (slide "Overlap or centroid?"):
/// for the A objects 1, 2, …, max label of `a`, the share that (i) has any pixel in B,
/// (ii) has more than half of its pixels in B, (iii) has its centroid in B. The centroid
/// (mean column, mean row of the object's pixels) is rounded to the nearest pixel.
/// Returns `(any, majority, centroid)`.
pub fn overlap_shares(a: &Image<u32>, b: &Image<bool>) -> (f32, f32, f32) {
    let n = a.pixels().iter().copied().max().unwrap_or(0) as usize;
    let mut area = vec![0.0f32; n + 1];
    let mut in_b = vec![0.0f32; n + 1];
    let mut sum_x = vec![0.0f32; n + 1];
    let mut sum_y = vec![0.0f32; n + 1];
    for (x, y) in a.coords() {
        let l = a[(x, y)] as usize;
        if l > 0 {
            area[l] += 1.0;
            sum_x[l] += x as f32;
            sum_y[l] += y as f32;
            if b[(x, y)] {
                in_b[l] += 1.0;
            }
        }
    }
    let (mut objects, mut any, mut majority, mut centroid) = (0, 0, 0, 0);
    for l in 1..=n {
        if area[l] == 0.0 {
            continue; // a label that does not occur
        }
        objects += 1;
        if in_b[l] > 0.0 {
            any += 1;
        }
        if in_b[l] > 0.5 * area[l] {
            majority += 1;
        }
        let (cx, cy) = ((sum_x[l] / area[l]).round() as usize, (sum_y[l] / area[l]).round() as usize);
        if b[(cx, cy)] {
            centroid += 1;
        }
    }
    let share = |k: i32| k as f32 / objects as f32;
    (share(any), share(majority), share(centroid))
}

/// C5a. The cross-K function at radius `r` (slide "The math: the cross-K function"):
/// K = area / (n_A n_B) · (number of pairs (A spot, B spot) at most r apart).
pub fn cross_k(a: &[Point], b: &[Point], area: f32, r: f32) -> f32 {
    let mut count = 0;
    for &pa in a {
        for &pb in b {
            if dist(pa, pb) <= r {
                count += 1;
            }
        }
    }
    area / (a.len() * b.len()) as f32 * count as f32
}

/// C5b. `n` random positions (in nm) inside the mask (slide "Randomisation for objects"):
/// draw a position uniformly in the whole image and keep it only if it lies in the mask.
pub fn relocate(n: usize, mask: &Image<bool>, rng: &mut Rng) -> Vec<Point> {
    let (w, h) = (mask.width() as f32 * PIXEL, mask.height() as f32 * PIXEL);
    let mut out = Vec::new();
    while out.len() < n {
        let p = (rng.uniform() * w, rng.uniform() * h);
        if in_mask(mask, p) {
            out.push(p);
        }
    }
    out
}
