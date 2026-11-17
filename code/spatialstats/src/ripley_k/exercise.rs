// Coding exercises for "Ripley's K, the L function and the pair correlation".
// Replace each `todo!()` with your code, then check it with `cargo test -p spatialstats ripley_k::`
// and look at `cargo run -p spatialstats -- ripley-k`.
#![allow(unused_variables, unused_imports)]

pub use crate::common::{csr, dist, hard_core, poisson, thomas, Point};

/// C1. The translation edge weight of a pair with difference `(dx, dy)` in the window
/// `[0, a] × [0, b]` (slide "The math: why the weights work"):
/// `|W| / |W ∩ (W + v)| = a b / ((a - |dx|) (b - |dy|))`.
pub fn translation_weight(dx: f64, dy: f64, a: f64, b: f64) -> f64 {
    todo!("C1: translation_weight")
}

/// C2a. The estimate of Ripley's K at distance `r` (slide "The math: the K function"):
/// `K = |W| / (n (n - 1)) · sum over ordered pairs i != j with d_ij <= r of w_ij`, with `w_ij` the
/// translation weight, or 1 without a correction.
///
/// Hint: two loops over all points (`enumerate` gives the index too), skip `i == j`. The
/// difference vector of a pair is `(q.0 - p.0, q.1 - p.1)`.
pub fn k_hat(points: &[Point], a: f64, b: f64, r: f64, translation: bool) -> f64 {
    todo!("C2a: k_hat")
}

/// C2b. `L(r) - r` with `L = sqrt(K / pi)`: 0 for CSR, above 0 clustered, below 0 regular.
pub fn l_minus_r(k: f64, r: f64) -> f64 {
    todo!("C2b: l_minus_r")
}

/// C3. The estimate of the pair correlation g at distance `r` (slide "The math: pair
/// correlation"), with the Epanechnikov kernel of half-width `h`,
/// `k_h(t) = 3 / (4h) · (1 - (t/h)²)` for `|t| <= h` (0 outside), and translation weights:
/// `g = |W| / (n (n - 1)) · sum over i != j of w_ij k_h(r - d_ij) / (2 pi d_ij)`.
///
/// Hint: the same double loop as `k_hat`; a pair contributes only if |r - d| <= h.
pub fn pair_correlation(points: &[Point], a: f64, b: f64, r: f64, h: f64) -> f64 {
    todo!("C3: pair_correlation")
}
