// Coding exercises for "Pretext tasks and masked image modelling". Replace each `todo!()` with your code,
// then check it with `cargo test -p selfsup pretext::` and look at `cargo run -p selfsup -- pretext`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{filters, linalg, Image};

/// A linear model of square windows: the mean window and the leading principal components
/// (unit vectors, row by row), with their variances (eigenvalues), largest first.
#[derive(Clone, Debug)]
pub struct WindowModel {
    pub side: usize,
    pub mean: Vec<f64>,
    pub comps: Vec<Vec<f64>>,
    pub variances: Vec<f64>,
}

/// C1. Fill the hidden pixels (`mask` = 1) by local interpolation: a Gaussian-weighted mean of
/// the visible pixels nearby (normalised convolution). Visible pixels are kept.
/// `blur(image × visible) / blur(visible)` at every hidden pixel.
///
/// Hint: `filters::gaussian_blur` twice; `img.map_xy(|x, y, v| ...)` to combine. Where the
/// blurred visibility is (almost) 0, use 0.5.
pub fn fill_interp(img: &Image<f32>, mask: &Image<f32>, sigma: f32) -> Image<f32> {
    todo!("C1: fill_interp")
}

/// C2. All `side × side` windows of the image with the given stride, each flattened row by
/// row into a vector.
///
/// Hint: two `while` loops over the top-left corner `(x0, y0)`, as long as the window fits.
pub fn windows(img: &Image<f32>, side: usize, stride: usize) -> Vec<Vec<f64>> {
    todo!("C2: windows")
}

/// C3. Learn the window model without labels: the mean of the windows, their covariance
/// matrix, and its `k` eigenvectors with the largest eigenvalues (PCA, the first day).
///
/// Hint: `linalg::sym_eigen(&cov)` returns `(values, vectors)`, largest first; keep the first `k`.
pub fn learn_model(windows: &[Vec<f64>], side: usize, k: usize) -> WindowModel {
    todo!("C3: learn_model")
}

/// C4. Complete one window: find the coefficients `c` of the components that best explain the
/// visible pixels (`visible[i]` true), with a ridge that keeps each coefficient within its
/// variance: minimise `Σ_visible (mean + Σ_a c_a u_a − window)² + Σ_a lambda c_a² / variance_a`.
/// Returns the whole fitted window `mean + Σ_a c_a u_a` (row by row).
///
/// Hint: the normal equations `A c = b` with `A[p][q] = Σ_visible u_p[i] u_q[i]` (plus
/// `lambda / variance_p` on the diagonal) and `b[p] = Σ_visible u_p[i] (window[i] − mean[i])`;
/// `linalg::solve(&a, &b)`. With `k = 0` the answer is the mean window.
pub fn complete_window(m: &WindowModel, window: &[f64], visible: &[bool], lambda: f64) -> Vec<f64> {
    todo!("C4: complete_window")
}

/// C5. The PSNR (peak 1) over the hidden pixels only.
///
/// Hint: the mean squared error over the pixels with `mask > 0.5`, then `10 log10(1 / mse)`.
pub fn hidden_psnr(result: &Image<f32>, truth: &Image<f32>, mask: &Image<f32>) -> f32 {
    todo!("C5: hidden_psnr")
}
