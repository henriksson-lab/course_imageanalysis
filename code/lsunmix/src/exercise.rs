// Coding exercises for "Least-squares unmixing". Replace each `todo!()` with your code,
// then check it with `cargo test -p lsunmix` and look at `cargo run -p lsunmix`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{linalg, Image};

/// A mixing matrix, row by row: `m[c][k]` = channel c of the spectrum of component k
/// (N rows = channels, K columns = components).
pub type Matrix = Vec<Vec<f32>>;

/// C1. Least-squares unmixing of one pixel by the normal equations: build `MᵀM` (K × K) and
/// `Mᵀy` (K numbers), then solve `MᵀM a = Mᵀy`.
///
/// Hint: `linalg::solve(&mtm, &mty)` solves the K × K system (in f64: build the matrices
/// as `Vec<Vec<f64>>` and `Vec<f64>`). Entry (i, j) of MᵀM is `Σ_c m[c][i] · m[c][j]`.
pub fn unmix_pixel(m: &Matrix, y: &[f32]) -> Vec<f32> {
    todo!("C1: unmix_pixel")
}

/// C2a. The pseudo-inverse `M⁺ = (MᵀM)⁻¹ Mᵀ` (K × N). Column c of M⁺ is the unmixing of a
/// pixel that is 1 in channel c and 0 elsewhere.
///
/// Hint: that is `unmix_pixel(m, e_c)` for the unit vector e_c, for every channel c.
pub fn pseudo_inverse(m: &Matrix) -> Matrix {
    todo!("C2a: pseudo_inverse")
}

/// C2b. Unmix a whole lambda stack (one image per channel) with one pseudo-inverse: abundance
/// map k is the weighted sum of the channels with the weights of row k of M⁺.
pub fn unmix_image(stack: &[Image<f32>], m: &Matrix) -> Vec<Image<f32>> {
    todo!("C2b: unmix_image")
}

/// C3a. The residual of one pixel: `r = y − M a`.
pub fn residual(m: &Matrix, y: &[f32], a: &[f32]) -> Vec<f32> {
    todo!("C3a: residual")
}

/// C3b. The residual map in units of the expected photon noise. With Poisson noise the
/// variance of a channel equals its expected count, so the squared residual should be about
/// as large as the fit, up to the factor (N − K)/N (the residual has only N − K free
/// directions). At every pixel, with `fit = M a`:
/// `sqrt( N/(N − K) · Σ_c r_c² / max(Σ_c fit_c, 1) )`. About 1 where the model explains
/// the pixel.
pub fn residual_map(stack: &[Image<f32>], m: &Matrix, abundances: &[Image<f32>]) -> Image<f32> {
    todo!("C3b: residual_map")
}

/// C4. The condition number κ = σ_max / σ_min of M: the singular values of M are the square
/// roots of the eigenvalues of MᵀM.
///
/// Hint: `linalg::sym_eigen(&mtm)` returns the eigenvalues of a symmetric matrix, largest
/// first.
pub fn condition_number(m: &Matrix) -> f32 {
    todo!("C4: condition_number")
}

/// C5. Weighted least squares of one pixel: solve `MᵀWM a = MᵀWy` with `W = diag(w)`
/// (slide "The math: WLS is maximum likelihood").
pub fn unmix_weighted(m: &Matrix, y: &[f32], w: &[f32]) -> Vec<f32> {
    todo!("C5: unmix_weighted")
}
