// Coding exercises for "Compressed sensing: fewer measurements than unknowns". Replace each
// `todo!()` with your code, then check it with `cargo test -p day11 compressed_sensing::` and look at
// `cargo run -p day11 -- compressed-sensing`.
// Vectors and matrices are f64, as in imgcourse::linalg; a matrix is `a[row][column]`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{fft, linalg, Complex, Rng};

/// The number of iterations that `recover` uses in `success_rate`.
pub const ITERATIONS: usize = 1000;

/// `A x`.
pub fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| row.iter().zip(x).map(|(r, v)| r * v).sum()).collect()
}

/// `Aᵀ r`.
pub fn mat_t_vec(a: &[Vec<f64>], r: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a[0].len()];
    for (row, ri) in a.iter().zip(r) {
        for (o, aij) in out.iter_mut().zip(row) {
            *o += aij * ri;
        }
    }
    out
}

/// L, the largest eigenvalue of `AᵀA`, by the power method (from the last unit).
pub fn largest_gain(a: &[Vec<f64>]) -> f64 {
    let n = a[0].len();
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut gain = 0.0;
    for _ in 0..100 {
        let w = mat_t_vec(a, &mat_vec(a, &v));
        gain = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        v = w.iter().map(|x| x / gain).collect();
    }
    gain
}

/// R1. The minimum-norm (least-energy) solution of the underdetermined `A x = y`:
/// `x = Aᵀ w` with `(A Aᵀ) w = y`.
///
/// Hint: `A Aᵀ` is `m × m`: its entry `(i, j)` is the dot product of rows `i` and `j` of `a`.
/// `linalg::solve(&aat, y)` solves the system (it returns an `Option`).
pub fn min_norm(a: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    todo!("R1: min_norm")
}

/// R2. The mutual coherence: the largest `|a_iᵀ a_j| / (‖a_i‖ ‖a_j‖)` over pairs of
/// different columns `i ≠ j`.
pub fn coherence(a: &[Vec<f64>]) -> f64 {
    todo!("R2: coherence")
}

/// R3. Zero filling: take the DFT of `x`, set the coefficients `k` with `keep[k] == false`
/// to 0, and transform back (the real part).
///
/// Hint: `fft::fft(&signal, false)` is the DFT of a `&[Complex]`, `fft::fft(&spectrum, true)`
/// the inverse. `Complex::from(v as f32)` makes a complex number from a real one; `c.re` is
/// the real part.
pub fn zero_filled(x: &[f64], keep: &[bool]) -> Vec<f64> {
    todo!("R3: zero_filled")
}

/// R4. Recover a sparse `x` from `y = A x`: FISTA (step 1/L, from the last unit) with λ
/// lowered step by step from λ_max to λ_max/10⁴ over the iterations, then a least-squares
/// refit on the support found, the entries larger than 1 % of the largest (this removes the
/// shrinkage bias).
///
/// Hint: λ_max is the largest `|(Aᵀy)_j|`. Multiply λ by `factor = (1e-4f64).powf(1.0 /
/// iterations as f64)` after every step. Refit as in OMP: `linalg::least_squares(&a_s, y)`
/// with only the support columns in `a_s`; skip the refit if the support is empty or has
/// more entries than there are measurements.
pub fn recover(a: &[Vec<f64>], y: &[f64], iterations: usize) -> Vec<f64> {
    todo!("R4: recover")
}

/// R5. One cell of the phase diagram: the fraction of `trials` random problems that
/// `recover` solves. Each trial: an `m × n` Gaussian matrix (entries of standard deviation
/// `1/√m`), a signal with `k` nonzeros of value ±1 at random positions, `y = A x`; success
/// if the relative error `‖x̂ − x‖ / ‖x‖` is below 1 %.
///
/// Hint: `let mut rng = Rng::new(seed);` then `rng.normal()` (an f32), `rng.below(n)` (a
/// position) and `rng.uniform() < 0.5` (a coin). Draw positions until `k` different ones
/// are filled. Use `recover(&a, &y, ITERATIONS)`.
pub fn success_rate(n: usize, m: usize, k: usize, trials: usize, seed: u64) -> f64 {
    todo!("R5: success_rate")
}
