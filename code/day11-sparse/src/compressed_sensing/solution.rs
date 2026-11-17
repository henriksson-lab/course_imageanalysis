// Model solution for the coding exercises of "Compressed sensing: fewer measurements than
// unknowns". Vectors and matrices are f64, as in imgcourse::linalg; a matrix is `a[row][column]`.

use imgcourse::{fft, linalg, Complex, Rng};

/// The number of iterations that `recover` uses in `success_rate`.
pub const ITERATIONS: usize = 1000;

/// `A x` and `Aᵀ r` (given, in `src/common.rs`).
pub use crate::common::{mat_t_vec, mat_vec};

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
pub fn min_norm(a: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let m = a.len();
    let mut aat = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            aat[i][j] = a[i].iter().zip(&a[j]).map(|(p, q)| p * q).sum();
        }
    }
    let w = linalg::solve(&aat, y).expect("A Aᵀ is singular");
    mat_t_vec(a, &w)
}

/// R2. The mutual coherence: the largest `|a_iᵀ a_j| / (‖a_i‖ ‖a_j‖)` over pairs of
/// different columns `i ≠ j`.
pub fn coherence(a: &[Vec<f64>]) -> f64 {
    let n = a[0].len();
    let column = |j: usize| -> Vec<f64> { a.iter().map(|row| row[j]).collect() };
    let dot = |p: &[f64], q: &[f64]| -> f64 { p.iter().zip(q).map(|(u, v)| u * v).sum() };
    let mut mu = 0.0f64;
    for i in 0..n {
        for j in i + 1..n {
            let (ci, cj) = (column(i), column(j));
            mu = mu.max(dot(&ci, &cj).abs() / (dot(&ci, &ci) * dot(&cj, &cj)).sqrt());
        }
    }
    mu
}

/// R3. Zero filling: take the DFT of `x`, set the coefficients `k` with `keep[k] == false`
/// to 0, and transform back (the real part).
pub fn zero_filled(x: &[f64], keep: &[bool]) -> Vec<f64> {
    let signal: Vec<Complex> = x.iter().map(|&v| Complex::from(v as f32)).collect();
    let mut spectrum = fft::fft(&signal, false);
    for k in 0..spectrum.len() {
        if !keep[k] {
            spectrum[k] = Complex::ZERO;
        }
    }
    fft::fft(&spectrum, true).iter().map(|c| c.re as f64).collect()
}

/// R4. Recover a sparse `x` from `y = A x`: FISTA (step 1/L, from the last unit) with λ
/// lowered step by step from λ_max to λ_max/10⁴ over the iterations, then a least-squares
/// refit on the support found, the entries larger than 1 % of the largest (this removes the
/// shrinkage bias).
pub fn recover(a: &[Vec<f64>], y: &[f64], iterations: usize) -> Vec<f64> {
    let n = a[0].len();
    let tau = 1.0 / largest_gain(a);
    let lambda_max = mat_t_vec(a, y).iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let factor = (1e-4f64).powf(1.0 / iterations as f64); // λ shrinks by this every step
    let mut lambda = lambda_max;
    let mut x = vec![0.0; n];
    let mut z = x.clone(); // FISTA: the step is taken from an extrapolated point
    let mut t = 1.0f64;
    for _ in 0..iterations {
        let az = mat_vec(a, &z);
        let r: Vec<f64> = y.iter().zip(&az).map(|(d, p)| d - p).collect();
        let g = mat_t_vec(a, &r);
        let mut x_new = vec![0.0; n];
        for j in 0..n {
            let v = z[j] + tau * g[j];
            let th = tau * lambda;
            x_new[j] = if v > th { v - th } else if v < -th { v + th } else { 0.0 };
        }
        let t_new = (1.0 + (1.0 + 4.0 * t * t).sqrt()) / 2.0;
        z = (0..n).map(|j| x_new[j] + (t - 1.0) / t_new * (x_new[j] - x[j])).collect();
        x = x_new;
        t = t_new;
        lambda *= factor;
    }
    // Refit on the support, if it is small enough to be fitted.
    let biggest = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let support: Vec<usize> = (0..n).filter(|&j| x[j].abs() > 0.01 * biggest).collect();
    if support.is_empty() || support.len() > a.len() {
        return x;
    }
    let a_s: Vec<Vec<f64>> = a.iter().map(|row| support.iter().map(|&j| row[j]).collect()).collect();
    if let Some(c) = linalg::least_squares(&a_s, y) {
        x = vec![0.0; n];
        for (k, &j) in support.iter().enumerate() {
            x[j] = c[k];
        }
    }
    x
}

/// R5. One cell of the phase diagram: the fraction of `trials` random problems that
/// `recover` solves. Each trial: an `m × n` Gaussian matrix (entries of standard deviation
/// `1/√m`), a signal with `k` nonzeros of value ±1 at random positions, `y = A x`; success
/// if the relative error `‖x̂ − x‖ / ‖x‖` is below 1 %.
pub fn success_rate(n: usize, m: usize, k: usize, trials: usize, seed: u64) -> f64 {
    let mut rng = Rng::new(seed);
    let mut successes = 0;
    for _ in 0..trials {
        let s = 1.0 / (m as f64).sqrt();
        let a: Vec<Vec<f64>> = (0..m).map(|_| (0..n).map(|_| s * rng.normal() as f64).collect()).collect();
        let mut x = vec![0.0; n];
        let mut placed = 0;
        while placed < k {
            let j = rng.below(n);
            if x[j] == 0.0 {
                x[j] = if rng.uniform() < 0.5 { 1.0 } else { -1.0 };
                placed += 1;
            }
        }
        let y = mat_vec(&a, &x);
        let xr = recover(&a, &y, ITERATIONS);
        let err: f64 = xr.iter().zip(&x).map(|(p, q)| (p - q).powi(2)).sum::<f64>().sqrt();
        if err < 0.01 * (k as f64).sqrt() {
            successes += 1;
        }
    }
    successes as f64 / trials as f64
}
