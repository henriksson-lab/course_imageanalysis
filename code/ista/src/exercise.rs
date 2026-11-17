// Coding exercises for "Solving with ℓ1: thresholding iterations and pursuit". Replace each
// `todo!()` with your code, then check it with `cargo test -p ista` and look at `cargo run -p ista`.
// Vectors and matrices are f64, as in imgcourse::linalg; a matrix is `a[row][column]`.
#![allow(unused_variables, unused_imports)]

use imgcourse::linalg;

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

/// The ℓ1 objective `½‖A x − y‖² + λ‖x‖₁`.
pub fn objective(a: &[Vec<f64>], y: &[f64], x: &[f64], lambda: f64) -> f64 {
    let misfit: f64 = mat_vec(a, x).iter().zip(y).map(|(p, d)| (p - d).powi(2)).sum();
    0.5 * misfit + lambda * x.iter().map(|v| v.abs()).sum::<f64>()
}

/// C1a. Soft-threshold every value at `t`: 0 if `|v| ≤ t`, otherwise move `t` towards 0.
pub fn soft_threshold(v: &[f64], t: f64) -> Vec<f64> {
    todo!("C1a: soft_threshold")
}

/// C1b. One ISTA iteration: a gradient (Landweber) step `z = x + τ Aᵀ(y − A x)`, then the
/// soft threshold of `z` at τλ.
///
/// Hint: `mat_vec` and `mat_t_vec` above do the matrix products.
pub fn ista_step(a: &[Vec<f64>], y: &[f64], x: &[f64], lambda: f64, tau: f64) -> Vec<f64> {
    todo!("C1b: ista_step")
}

/// C2. ISTA: start at 0 and repeat `ista_step` `iterations` times.
///
/// Hint: the number of unknowns is the number of columns, `a[0].len()`.
pub fn ista(a: &[Vec<f64>], y: &[f64], lambda: f64, tau: f64, iterations: usize) -> Vec<f64> {
    todo!("C2: ista")
}

/// C3. L, the largest gain of `AᵀA` (its largest eigenvalue), by the power method: start
/// from any vector, apply `AᵀA` and rescale to length 1, `iterations` times. The safe ISTA
/// step is τ = 1/L.
///
/// Hint: for a unit vector `v` close to the top eigenvector, the length of `AᵀA v` is L.
pub fn largest_gain(a: &[Vec<f64>], iterations: usize) -> f64 {
    todo!("C3: largest_gain")
}

/// C4. FISTA: ISTA steps taken from an extrapolated point. With `t = 1` at the start:
/// `x_new = ista_step(z)`, `t_new = (1 + √(1 + 4t²)) / 2`,
/// `z = x_new + (t − 1)/t_new · (x_new − x)`.
pub fn fista(a: &[Vec<f64>], y: &[f64], lambda: f64, tau: f64, iterations: usize) -> Vec<f64> {
    todo!("C4: fista")
}

/// C5. Orthogonal matching pursuit: pick the column most correlated with the residual
/// (`|a_jᵀ r| / ‖a_j‖`), refit all chosen columns by least squares, update the residual;
/// stop after `max_atoms` columns or when `‖r‖² ≤ tolerance`. Returns the full coefficient
/// vector (zeros for the columns that were not chosen).
///
/// Hint: `linalg::least_squares(&a_s, y)` fits `a_s c ≈ y`, where `a_s` holds only the chosen
/// columns: `a_s[i][k] = a[i][chosen[k]]`. It returns an `Option` (`None` if the chosen
/// columns are linearly dependent; then stop).
pub fn omp(a: &[Vec<f64>], y: &[f64], max_atoms: usize, tolerance: f64) -> Vec<f64> {
    todo!("C5: omp")
}
