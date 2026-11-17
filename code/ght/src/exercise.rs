// Coding exercises for "The generalised Hough transform".
// Replace each `todo!()` with your code, then check it with `cargo test -p ght`
// and look at `cargo run -p ght`.
#![allow(unused_variables, unused_imports)]

use crate::sim::EdgePoint;
use imgcourse::Image;
use std::f32::consts::TAU;

/// The R-table: for each direction bin, the vectors from edge points to the reference point.
pub type RTable = Vec<Vec<[f32; 2]>>;

/// C1a. The direction bin of a gradient direction `theta` (radians), with `n_bins` bins of
/// width 360°/n_bins centred on 0°, 360°/n_bins, …: bin 0 holds the directions closest to 0°.
///
/// Hint: divide θ by the bin width, round, and wrap with `rem_euclid(n_bins as isize)`.
pub fn bin_of(theta: f32, n_bins: usize) -> usize {
    todo!("C1a: bin_of")
}

/// C1b. Build the R-table of a template: for every template edge point, store the vector
/// `reference − point` in the bin of its gradient direction.
///
/// Hint: start from `vec![Vec::new(); n_bins]`.
pub fn build_r_table(template: &[EdgePoint], reference: [f32; 2], n_bins: usize) -> RTable {
    todo!("C1b: build_r_table")
}

/// C2. Vote with the R-table (translation only): every edge point looks up the bin of its
/// direction and adds one vote at `point + v` for every vector `v` there (rounded to the
/// nearest pixel, if inside the image).
///
/// Hint: this is `vote_pose` with φ = 0 and scale 1; or write the loop directly.
pub fn vote(table: &RTable, edges: &[EdgePoint], width: usize, height: usize) -> Image<f32> {
    todo!("C2: vote")
}

/// C3. Vote for the pose rotated by `phi` and scaled by `scale`: an edge point with direction θ
/// looks up the bin of `θ − phi` and votes at `point + scale · R(phi) v` for every vector `v`
/// there.
///
/// Hint: with `c = cos φ`, `s = sin φ` the rotated, scaled vector is
/// `(scale (c v0 − s v1), scale (s v0 + c v1))`.
pub fn vote_pose(table: &RTable, edges: &[EdgePoint], width: usize, height: usize, phi: f32, scale: f32) -> Image<f32> {
    todo!("C3: vote_pose")
}
