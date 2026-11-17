// Coding exercises for "Objects instead of pixels". Replace each `todo!()` with your code,
// then check it with `cargo test -p day44 coloc_objects::` and look at `cargo run -p day44 -- coloc-objects`.
#![allow(unused_variables, unused_imports)]

use crate::coloc_objects::sim::{dist, in_mask, Point, PIXEL};
use imgcourse::{Image, Rng};

/// C1a. For every A spot, the index of the nearest B spot and the distance to it
/// (slide "Distances between channels"). `b` is not empty.
///
/// Hint: `dist(p, q)` (given) is the distance between two points.
pub fn nearest(a: &[Point], b: &[Point]) -> Vec<(usize, f32)> {
    todo!("C1a: nearest")
}

/// C1b. The share of A spots whose nearest B spot is at most `d_max` away.
pub fn fraction_within(a: &[Point], b: &[Point], d_max: f32) -> f32 {
    todo!("C1b: fraction_within")
}

/// C2. One-to-one pairing: the number of pairs (A spot, B spot) at most `d_max` apart when
/// every spot may be used only once. Take the closest pairs first (greedy): list all pairs
/// within `d_max`, sort them by distance, and accept a pair if neither spot is used yet.
pub fn one_to_one(a: &[Point], b: &[Point], d_max: f32) -> usize {
    todo!("C2: one_to_one")
}

/// C3a. The chance that a random point has a B spot within `d`, for B spots placed at random
/// with density `rho` (spots per unit area): 1 − exp(−rho π d²) (slide "The chance level").
pub fn chance_level(rho: f32, d: f32) -> f32 {
    todo!("C3a: chance_level")
}

/// C3b. The fraction corrected for chance: (f_obs − f_chance) / (1 − f_chance).
pub fn corrected(f_obs: f32, f_chance: f32) -> f32 {
    todo!("C3b: corrected")
}

/// C4. The three overlap definitions for labelled objects (slide "Overlap or centroid?"):
/// for the A objects 1, 2, …, max label of `a`, the share that (i) has any pixel in B,
/// (ii) has more than half of its pixels in B, (iii) has its centroid in B. The centroid
/// (mean column, mean row of the object's pixels) is rounded to the nearest pixel.
/// Returns `(any, majority, centroid)`.
///
/// Hint: one pass over the image collects, per label, the area, the number of pixels in B
/// and the sums of x and of y (vectors indexed by the label). Then one loop over the labels.
pub fn overlap_shares(a: &Image<u32>, b: &Image<bool>) -> (f32, f32, f32) {
    todo!("C4: overlap_shares")
}

/// C5a. The cross-K function at radius `r` (slide "The math: the cross-K function"):
/// K = area / (n_A n_B) · (number of pairs (A spot, B spot) at most r apart).
pub fn cross_k(a: &[Point], b: &[Point], area: f32, r: f32) -> f32 {
    todo!("C5a: cross_k")
}

/// C5b. `n` random positions (in nm) inside the mask (slide "Randomisation for objects"):
/// draw a position uniformly in the whole image and keep it only if it lies in the mask.
///
/// Hint: the image is `mask.width() * PIXEL` nm wide; `in_mask(mask, p)` (given) says
/// whether a point in nm lies in the mask.
pub fn relocate(n: usize, mask: &Image<bool>, rng: &mut Rng) -> Vec<Point> {
    todo!("C5b: relocate")
}
