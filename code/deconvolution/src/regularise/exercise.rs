// Coding exercises for "Regularisation: what we add, and how much". Replace each `todo!()`
// with your code, then check it with `cargo test -p deconvolution regularise::` and look at `cargo run -p deconvolution -- regularise`.
#![allow(unused_variables, unused_imports)]
// Everything is 1D: a profile of n samples, the blur a matrix A, in f64.

use imgcourse::linalg::solve;

/// What the penalty measures: the size of the values (`Energy`, D = identity) or the
/// roughness (`Roughness`, D = second differences `x[i−1] − 2 x[i] + x[i+1]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Penalty {
    Energy,
    Roughness,
}

/// C1. The blur of a profile of `n` samples by a Gaussian of standard deviation `sigma`, as an
/// `n × n` matrix: row `i` holds the weights `exp(−(i − j)² / 2σ²)` for `|i − j| ≤ ceil(3σ)`,
/// divided by their sum (so that a constant profile stays constant, also at the ends).
///
/// Hint: `vec![vec![0.0; n]; n]`; for row `i`, `j` runs over `max(i − r, 0) ..= min(i + r, n − 1)`.
pub fn blur_matrix(n: usize, sigma: f64) -> Vec<Vec<f64>> {
    todo!("C1: blur_matrix")
}

/// The penalty matrix D for `n` samples: the identity, or one row per second difference
/// (`n − 2` rows). Given (not an exercise).
pub fn penalty_matrix(n: usize, penalty: Penalty) -> Vec<Vec<f64>> {
    match penalty {
        Penalty::Energy => (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect(),
        Penalty::Roughness => (1..n.saturating_sub(1))
            .map(|i| {
                let mut row = vec![0.0; n];
                (row[i - 1], row[i], row[i + 1]) = (1.0, -2.0, 1.0);
                row
            })
            .collect(),
    }
}

/// C2. Tikhonov regularisation: the `x` that minimises `‖A x − y‖² + λ ‖D x‖²`, found by
/// solving the normal equations `(AᵀA + λ DᵀD) x = Aᵀy`.
///
/// Hint: `(AᵀA)[i][j]` is the sum over the rows of A of `row[i] · row[j]`; the same for D
/// (`penalty_matrix`). Then `solve(&m, &rhs)` returns `Option<Vec<f64>>`.
pub fn tikhonov(a: &[Vec<f64>], y: &[f64], lambda: f64, penalty: Penalty) -> Vec<f64> {
    todo!("C2: tikhonov")
}

/// C3a. Tikhonov's filter factor: the fraction `s² / (s² + λ)` of a component with
/// singular value (gain) `s` that the restoration keeps.
pub fn tikhonov_factor(s: f64, lambda: f64) -> f64 {
    todo!("C3a: tikhonov_factor")
}

/// C3b. Landweber's filter factor after `k` iterations (step 1): `1 − (1 − s²)^k`.
pub fn landweber_factor(s: f64, k: u32) -> f64 {
    todo!("C3b: landweber_factor")
}

/// C4. Landweber iteration, starting from zero: `k` times
/// `x ← x + ω Aᵀ(y − A x)` (blur the estimate, compare with the data, blur the difference back,
/// add).
///
/// Hint: first the residual `r = y − A x` (one number per row of A), then `x[j] += ω · (sum over rows i of A[i][j] · r[i])`.
pub fn landweber(a: &[Vec<f64>], y: &[f64], k: u32, omega: f64) -> Vec<f64> {
    todo!("C4: landweber")
}

/// C5a. The root-mean-square residual `sqrt(‖A x − y‖² / m)`, with `m` data values.
pub fn rms_residual(a: &[Vec<f64>], x: &[f64], y: &[f64]) -> f64 {
    todo!("C5a: rms_residual")
}

/// C5b. The discrepancy principle: the λ (between 10⁻⁸ and 10⁴) at which the RMS residual of
/// the Tikhonov estimate equals the noise level `sigma`. The residual grows with λ, so halve
/// the interval of `log10 λ` 40 times (bisection).
///
/// Hint: keep an interval `lo..hi` of `log10 λ`; at its middle compute the Tikhonov estimate and
/// its residual. Below `sigma`: the answer is larger (move `lo` up); else move `hi` down.
pub fn discrepancy_lambda(a: &[Vec<f64>], y: &[f64], sigma: f64, penalty: Penalty) -> f64 {
    todo!("C5b: discrepancy_lambda")
}
