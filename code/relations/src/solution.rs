// Model solution for the coding exercises of "Neighbours, parents and children".

use imgcourse::Image;
use std::collections::BTreeSet;

/// The 4 neighbours and the 8 neighbours of a pixel, as offsets `(dx, dy)`.
pub const N4: [(isize, isize); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
pub const N8: [(isize, isize); 8] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)];

/// The label at `(x, y)`, or 0 (background) outside the image.
pub fn label_at(labels: &Image<u32>, x: isize, y: isize) -> u32 {
    if labels.contains(x, y) { labels[(x as usize, y as usize)] } else { 0 }
}

/// C1. The neighbours of every object: `result[l]` is the set of labels adjacent to label `l`
/// (4- or 8-neighbouring pixels, `eight` chooses). `result[0]` stays empty; the result has
/// one entry per label from 0 to the largest label.
pub fn neighbours(labels: &Image<u32>, eight: bool) -> Vec<BTreeSet<u32>> {
    let max = labels.pixels().iter().copied().max().unwrap_or(0) as usize;
    let mut result = vec![BTreeSet::new(); max + 1];
    let offsets: &[(isize, isize)] = if eight { &N8 } else { &N4 };
    for (x, y) in labels.coords() {
        let a = labels[(x, y)];
        if a == 0 {
            continue;
        }
        for &(dx, dy) in offsets {
            let b = label_at(labels, x as isize + dx, y as isize + dy);
            if b != 0 && b != a {
                result[a as usize].insert(b);
            }
        }
    }
    result
}

/// C2. Percent touching of object `label` (8-neighbours): of its boundary pixels (those with
/// a neighbour of another value, outside the image counting as background), the percentage
/// that have a neighbour belonging to another object.
pub fn percent_touching(labels: &Image<u32>, label: u32) -> f32 {
    let (mut boundary, mut touching) = (0, 0);
    for (x, y) in labels.coords() {
        if labels[(x, y)] != label {
            continue;
        }
        let around: Vec<u32> = N8.iter().map(|&(dx, dy)| label_at(labels, x as isize + dx, y as isize + dy)).collect();
        if around.iter().any(|&l| l != label) {
            boundary += 1;
            if around.iter().any(|&l| l != label && l != 0) {
                touching += 1;
            }
        }
    }
    100.0 * touching as f32 / boundary as f32
}

/// C3a. For every point: the indices of its closest and second closest other points, and the
/// two distances: `(first, d1, second, d2)`. Needs at least 3 points.
pub fn closest_two(points: &[[f32; 2]]) -> Vec<(usize, f32, usize, f32)> {
    let dist = |a: [f32; 2], b: [f32; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    let mut out = Vec::new();
    for (i, &p) in points.iter().enumerate() {
        let (mut first, mut d1, mut second, mut d2) = (0, f32::INFINITY, 0, f32::INFINITY);
        for (j, &q) in points.iter().enumerate() {
            if j == i {
                continue;
            }
            let d = dist(p, q);
            if d < d1 {
                (second, d2) = (first, d1);
                (first, d1) = (j, d);
            } else if d < d2 {
                (second, d2) = (j, d);
            }
        }
        out.push((first, d1, second, d2));
    }
    out
}

/// C3b. The angle (in degrees, 0 to 180) at point `p` between the directions to `a` and `b`.
pub fn angle_at(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let (u, v) = ([a[0] - p[0], a[1] - p[1]], [b[0] - p[0], b[1] - p[1]]);
    let cos = (u[0] * v[0] + u[1] * v[1]) / ((u[0] * u[0] + u[1] * u[1]).sqrt() * (v[0] * v[0] + v[1] * v[1]).sqrt());
    cos.clamp(-1.0, 1.0).acos().to_degrees()
}

/// C3c. The Clark–Evans ratio: the mean nearest-neighbour distance divided by `0.5 / sqrt(ρ)`,
/// with `ρ` = number of points / `area`.
pub fn clark_evans(points: &[[f32; 2]], area: f32) -> f32 {
    let mean_nn = closest_two(points).iter().map(|c| c.1).sum::<f32>() / points.len() as f32;
    let rho = points.len() as f32 / area;
    mean_nn / (0.5 / rho.sqrt())
}

/// C4. "Expand until adjacent": give every background pixel the label of the nearest object
/// pixel (Euclidean distance between pixel centres; on a tie, the first one found).
pub fn expand_labels(labels: &Image<u32>) -> Image<u32> {
    let objects: Vec<(f32, f32, u32)> = labels.coords().filter(|&p| labels[p] != 0).map(|(x, y)| (x as f32, y as f32, labels[(x, y)])).collect();
    labels.map_xy(|x, y, l| {
        if l != 0 {
            return l;
        }
        let (mut best, mut best_d2) = (0, f32::INFINITY);
        for &(ox, oy, ol) in &objects {
            let d2 = (x as f32 - ox).powi(2) + (y as f32 - oy).powi(2);
            if d2 < best_d2 {
                (best, best_d2) = (ol, d2);
            }
        }
        best
    })
}

/// C5a. The parent of every child by largest overlap: `result[c]` = `(parent, fraction)`, the
/// parent label covering most pixels of child `c` and the fraction of the child's pixels it
/// covers; `(0, 0.0)` if the child lies in no parent. Index 0 (background) is unused.
pub fn parent_by_overlap(children: &Image<u32>, parents: &Image<u32>) -> Vec<(u32, f32)> {
    let nc = children.pixels().iter().copied().max().unwrap_or(0) as usize;
    let np = parents.pixels().iter().copied().max().unwrap_or(0) as usize;
    let mut table = vec![vec![0usize; np + 1]; nc + 1]; // table[child][parent] = pixels in common
    for (x, y) in children.coords() {
        table[children[(x, y)] as usize][parents[(x, y)] as usize] += 1;
    }
    let mut result = vec![(0, 0.0); nc + 1];
    for c in 1..=nc {
        let size: usize = table[c].iter().sum();
        for p in 1..=np {
            if table[c][p] > 0 && (result[c].0 == 0 || table[c][p] > table[c][result[c].0 as usize]) {
                result[c] = (p as u32, table[c][p] as f32 / size as f32);
            }
        }
    }
    result
}

/// C5b. The parent of every child by centroid: the parent label at the child's centroid,
/// rounded to the nearest pixel (0 if that is background). Index 0 is unused.
pub fn parent_by_centroid(children: &Image<u32>, parents: &Image<u32>) -> Vec<u32> {
    let nc = children.pixels().iter().copied().max().unwrap_or(0) as usize;
    let mut sum = vec![(0.0f32, 0.0f32, 0.0f32); nc + 1];
    for (x, y) in children.coords() {
        let c = children[(x, y)] as usize;
        sum[c] = (sum[c].0 + x as f32, sum[c].1 + y as f32, sum[c].2 + 1.0);
    }
    let mut result = vec![0; nc + 1];
    for c in 1..=nc {
        let (sx, sy, n) = sum[c];
        if n > 0.0 {
            result[c] = parents[((sx / n).round() as usize, (sy / n).round() as usize)];
        }
    }
    result
}
