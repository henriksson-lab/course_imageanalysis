// Coding exercises for "Neighbours, parents and children". Replace each `todo!()` with your
// code, then check it with `cargo test -p measurements relations::` and look at `cargo run -p measurements -- relations`.
#![allow(unused_variables, unused_imports)]

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
///
/// Hint: `vec![BTreeSet::new(); n]` makes `n` empty sets; `set.insert(b)` ignores repeats.
pub fn neighbours(labels: &Image<u32>, eight: bool) -> Vec<BTreeSet<u32>> {
    todo!("C1: neighbours")
}

/// C2. Percent touching of object `label` (8-neighbours): of its boundary pixels (those with
/// a neighbour of another value, outside the image counting as background), the percentage
/// that have a neighbour belonging to another object.
pub fn percent_touching(labels: &Image<u32>, label: u32) -> f32 {
    todo!("C2: percent_touching")
}

/// C3a. For every point: the indices of its closest and second closest other points, and the
/// two distances: `(first, d1, second, d2)`. Needs at least 3 points.
///
/// Hint: keep the best two while looping over the other points; a new closest one pushes the
/// old closest down to second place.
pub fn closest_two(points: &[[f32; 2]]) -> Vec<(usize, f32, usize, f32)> {
    todo!("C3a: closest_two")
}

/// C3b. The angle (in degrees, 0 to 180) at point `p` between the directions to `a` and `b`.
///
/// Hint: the cosine of the angle is the dot product of the two directions divided by their
/// lengths; `cos.clamp(-1.0, 1.0).acos().to_degrees()`.
pub fn angle_at(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    todo!("C3b: angle_at")
}

/// C3c. The Clark–Evans ratio: the mean nearest-neighbour distance divided by `0.5 / sqrt(ρ)`,
/// with `ρ` = number of points / `area`.
pub fn clark_evans(points: &[[f32; 2]], area: f32) -> f32 {
    todo!("C3c: clark_evans")
}

/// C4. "Expand until adjacent": give every background pixel the label of the nearest object
/// pixel (Euclidean distance between pixel centres; on a tie, the first one found).
///
/// Hint: first make a list of all object pixels with their labels; then, for every
/// background pixel, search that list. Comparing squared distances is enough.
pub fn expand_labels(labels: &Image<u32>) -> Image<u32> {
    todo!("C4: expand_labels")
}

/// C5a. The parent of every child by largest overlap: `result[c]` = `(parent, fraction)`, the
/// parent label covering most pixels of child `c` and the fraction of the child's pixels it
/// covers; `(0, 0.0)` if the child lies in no parent. Index 0 (background) is unused.
///
/// Hint: one pass over the pixels fills a table `table[child][parent]` of pixel counts
/// (`vec![vec![0usize; np + 1]; nc + 1]`); then read each child's row, skipping parent 0.
pub fn parent_by_overlap(children: &Image<u32>, parents: &Image<u32>) -> Vec<(u32, f32)> {
    todo!("C5a: parent_by_overlap")
}

/// C5b. The parent of every child by centroid: the parent label at the child's centroid,
/// rounded to the nearest pixel (0 if that is background). Index 0 is unused.
pub fn parent_by_centroid(children: &Image<u32>, parents: &Image<u32>) -> Vec<u32> {
    todo!("C5b: parent_by_centroid")
}
