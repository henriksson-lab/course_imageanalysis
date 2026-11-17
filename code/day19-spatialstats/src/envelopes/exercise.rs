// Coding exercises for "Is it more than chance? Simulation envelopes".
// Replace each `todo!()` with your code, then check it with `cargo test -p day19 envelopes::`
// and look at `cargo run -p day19 -- envelopes`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Rng;

pub use crate::common::{csr, dist, hard_core, poisson, thomas, Point};

/// `L(r) - r` with the translation correction (the ripley-k unit) at every radius of `radii`,
/// from one pass over the pairs. Given.
pub fn l_curve(points: &[Point], a: f64, b: f64, radii: &[f64]) -> Vec<f64> {
    let n = points.len() as f64;
    let mut sums = vec![0.0; radii.len()];
    for (i, &p) in points.iter().enumerate() {
        for &q in &points[i + 1..] {
            let d = dist(p, q);
            let w = a * b / ((a - (q.0 - p.0).abs()) * (b - (q.1 - p.1).abs()));
            for (k, &r) in radii.iter().enumerate() {
                if d <= r {
                    sums[k] += 2.0 * w; // the pair counts in both orders
                }
            }
        }
    }
    radii.iter().zip(&sums).map(|(&r, &s)| (a * b / (n * (n - 1.0)) * s / std::f64::consts::PI).sqrt() - r).collect()
}

/// C1. The one-sided Monte Carlo p-value (slide "The Monte Carlo p-value"):
/// `(1 + number of simulated values at least as extreme as the observed) / (1 + s)`.
/// "As extreme" means `>=` the observed value when `larger` is true, `<=` otherwise.
///
/// Hint: `simulated.iter().filter(|&&t| ...).count()`.
pub fn mc_p_value(observed: f64, simulated: &[f64], larger: bool) -> f64 {
    todo!("C1: mc_p_value")
}

/// C2a. The pointwise envelope: at every radius (index `k`), the smallest and the largest
/// value of the simulated curves. Returns `(lower, upper)`.
///
/// Hint: start `lower` at `f64::INFINITY` and `upper` at `f64::NEG_INFINITY` for every radius.
pub fn pointwise_envelope(curves: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    todo!("C2a: pointwise_envelope")
}

/// C2b. The indices of the radii at which the observed curve lies outside the envelope.
pub fn escapes(observed: &[f64], lower: &[f64], upper: &[f64]) -> Vec<usize> {
    todo!("C2b: escapes")
}

/// C3. The MAD test (slide "A global test: the MAD test"): `T = max over r of |curve(r)|` for
/// the observed curve and for every simulated one (the curves are deviations from the CSR
/// value, such as L(r) - r), and the Monte Carlo p-value of the observed T (large T is extreme).
/// Returns `(T_observed, p)`.
///
/// Hint: T of one curve is `c.iter().fold(0.0f64, |m, v| m.max(v.abs()))`; then `mc_p_value`
/// with `larger = true`.
pub fn mad_test(observed: &[f64], simulated: &[Vec<f64>]) -> (f64, f64) {
    todo!("C3: mad_test")
}

/// C4. How often is CSR rejected? Simulate `patterns` CSR patterns of `n` points in the window;
/// for each, compute its `L(r) - r` curve and those of `s` further CSR patterns (the null), and
/// record whether it leaves the pointwise envelope at some radius, and whether the MAD test
/// gives p <= 0.05. Returns the two shares `(pointwise, mad)`.
///
/// Hint: `l_curve` (given), `csr`, and your C2 and C3. Draw the observed pattern and the s
/// simulations from the same `rng`.
pub fn false_alarm_rates(n: usize, a: f64, b: f64, radii: &[f64], s: usize, patterns: usize, rng: &mut Rng) -> (f64, f64) {
    todo!("C4: false_alarm_rates")
}
