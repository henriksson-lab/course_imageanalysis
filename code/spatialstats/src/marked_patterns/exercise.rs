// Coding exercises for "Two cell types and an uneven density".
// Replace each `todo!()` with your code, then check it with `cargo test -p spatialstats marked_patterns::`
// and look at `cargo run -p spatialstats -- marked-patterns`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Rng;

pub use crate::common::{csr, dist, hard_core, normal_cdf, poisson, thomas, Point};

/// The translation edge weight `a b / ((a - |dx|)(b - |dy|))` (the ripley-k unit). Given.
pub fn translation_weight(dx: f64, dy: f64, a: f64, b: f64) -> f64 {
    a * b / ((a - dx.abs()) * (b - dy.abs()))
}

/// C1. The cross-K function (slide "The math: cross-K"): type-1 points around type-0 points,
/// `K01 = |W| / (n0 n1) · sum over i in type 0, j in type 1 with d_ij <= r of w_ij`,
/// with the translation weight `w_ij`, or 1 without a correction.
///
/// Hint: one loop over the type-0 points, one over the type-1 points; `translation_weight` is
/// given.
pub fn cross_k(type0: &[Point], type1: &[Point], a: f64, b: f64, r: f64, translation: bool) -> f64 {
    todo!("C1: cross_k")
}

/// C2. Random labelling: the same labels in a random order (Fisher–Yates shuffle: for k from
/// the last index down to 1, swap element k with a random element 0..=k).
///
/// Hint: `rng.below(k + 1)` gives a whole number in 0..=k; `out.swap(k, j)`.
pub fn permute_labels(labels: &[u8], rng: &mut Rng) -> Vec<u8> {
    todo!("C2: permute_labels")
}

/// C3. The mark connection function `p01(r)` (slide "The mark connection function"): of all
/// pairs of points whose distance lies within `h` of `r`, the share whose two labels differ.
/// NaN if there is no such pair.
///
/// Hint: loop over the pairs `i < j` only, so that each pair counts once.
pub fn mark_connection(points: &[Point], labels: &[u8], r: f64, h: f64) -> f64 {
    todo!("C3: mark_connection")
}

/// C4. The inhomogeneous K (slide "The math: the inhomogeneous K") with translation weights:
/// `K = 1/|W| · sum over i != j with d_ij <= r of w_ij / (rho_i rho_j)`, where `rho_i` is the
/// leave-one-out kernel estimate of the intensity at point i: the sum of 2D Gaussians
/// (standard deviation `sigma`, integral 1) of all **other** points, divided by the edge
/// correction `e = [Φ((a - x)/σ) - Φ(-x/σ)] · [Φ((b - y)/σ) - Φ(-y/σ)]`.
///
/// Hint: first the vector of the n intensities `rho[i]` (a loop over the other points, then
/// divide by e), then the double loop over the pairs as in Ripley's K.
pub fn k_inhom(points: &[Point], a: f64, b: f64, r: f64, sigma: f64) -> f64 {
    todo!("C4: k_inhom")
}
