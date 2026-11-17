// Model solution for the coding exercises of "Regularisation: what we add, and how much".
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
pub fn blur_matrix(n: usize, sigma: f64) -> Vec<Vec<f64>> {
    let r = (3.0 * sigma).ceil() as isize;
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        let mut total = 0.0;
        for j in (i as isize - r).max(0)..=(i as isize + r).min(n as isize - 1) {
            let d = (i as isize - j) as f64;
            a[i][j as usize] = (-d * d / (2.0 * sigma * sigma)).exp();
            total += a[i][j as usize];
        }
        for v in a[i].iter_mut() {
            *v /= total;
        }
    }
    a
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
pub fn tikhonov(a: &[Vec<f64>], y: &[f64], lambda: f64, penalty: Penalty) -> Vec<f64> {
    let n = a[0].len();
    let d = penalty_matrix(n, penalty);
    let mut m = vec![vec![0.0; n]; n];
    let mut rhs = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            let ata: f64 = a.iter().map(|row| row[i] * row[j]).sum();
            let dtd: f64 = d.iter().map(|row| row[i] * row[j]).sum();
            m[i][j] = ata + lambda * dtd;
        }
        rhs[i] = a.iter().zip(y).map(|(row, yk)| row[i] * yk).sum();
    }
    solve(&m, &rhs).expect("the normal equations are singular: is λ > 0?")
}

/// C3a. Tikhonov's filter factor: the fraction `s² / (s² + λ)` of a component with
/// singular value (gain) `s` that the restoration keeps.
pub fn tikhonov_factor(s: f64, lambda: f64) -> f64 {
    s * s / (s * s + lambda)
}

/// C3b. Landweber's filter factor after `k` iterations (step 1): `1 − (1 − s²)^k`.
pub fn landweber_factor(s: f64, k: u32) -> f64 {
    1.0 - (1.0 - s * s).powi(k as i32)
}

/// C4. Landweber iteration, starting from zero: `k` times
/// `x ← x + ω Aᵀ(y − A x)` (blur the estimate, compare with the data, blur the difference back,
/// add).
pub fn landweber(a: &[Vec<f64>], y: &[f64], k: u32, omega: f64) -> Vec<f64> {
    let n = a[0].len();
    let mut x = vec![0.0; n];
    for _ in 0..k {
        // r = y − A x
        let r: Vec<f64> = a.iter().zip(y).map(|(row, yi)| yi - row.iter().zip(&x).map(|(aij, xj)| aij * xj).sum::<f64>()).collect();
        // x = x + ω Aᵀ r
        for j in 0..n {
            x[j] += omega * a.iter().zip(&r).map(|(row, ri)| row[j] * ri).sum::<f64>();
        }
    }
    x
}

/// C5a. The root-mean-square residual `sqrt(‖A x − y‖² / m)`, with `m` data values.
pub fn rms_residual(a: &[Vec<f64>], x: &[f64], y: &[f64]) -> f64 {
    let sum: f64 = a.iter().zip(y).map(|(row, yi)| (row.iter().zip(x).map(|(aij, xj)| aij * xj).sum::<f64>() - yi).powi(2)).sum();
    (sum / y.len() as f64).sqrt()
}

/// C5b. The discrepancy principle: the λ (between 10⁻⁸ and 10⁴) at which the RMS residual of
/// the Tikhonov estimate equals the noise level `sigma`. The residual grows with λ, so halve
/// the interval of `log10 λ` 40 times (bisection).
pub fn discrepancy_lambda(a: &[Vec<f64>], y: &[f64], sigma: f64, penalty: Penalty) -> f64 {
    let (mut lo, mut hi) = (-8.0f64, 4.0f64);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        let lambda = 10f64.powf(mid);
        if rms_residual(a, &tikhonov(a, y, lambda, penalty), y) < sigma {
            lo = mid; // still fitting the noise: more regularisation
        } else {
            hi = mid;
        }
    }
    10f64.powf(0.5 * (lo + hi))
}
