// Model solution for the coding exercises of "Pretext tasks and masked image modelling".

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
pub fn fill_interp(img: &Image<f32>, mask: &Image<f32>, sigma: f32) -> Image<f32> {
    let visible = mask.map(|m| 1.0 - m);
    let num = filters::gaussian_blur(&img.zip_map(&visible, |v, w| v * w), sigma);
    let den = filters::gaussian_blur(&visible, sigma);
    img.map_xy(|x, y, v| if mask[(x, y)] > 0.5 { if den[(x, y)] > 1e-6 { num[(x, y)] / den[(x, y)] } else { 0.5 } } else { v })
}

/// C2. All `side × side` windows of the image with the given stride, each flattened row by
/// row into a vector.
pub fn windows(img: &Image<f32>, side: usize, stride: usize) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    let mut y0 = 0;
    while y0 + side <= img.height() {
        let mut x0 = 0;
        while x0 + side <= img.width() {
            let mut v = Vec::with_capacity(side * side);
            for y in 0..side {
                for x in 0..side {
                    v.push(img[(x0 + x, y0 + y)] as f64);
                }
            }
            out.push(v);
            x0 += stride;
        }
        y0 += stride;
    }
    out
}

/// C3. Learn the window model without labels: the mean of the windows, their covariance
/// matrix, and its `k` eigenvectors with the largest eigenvalues (PCA, the first day).
pub fn learn_model(windows: &[Vec<f64>], side: usize, k: usize) -> WindowModel {
    let (n, d) = (windows.len() as f64, side * side);
    let mut mean = vec![0.0; d];
    for w in windows {
        for i in 0..d {
            mean[i] += w[i] / n;
        }
    }
    let mut cov = vec![vec![0.0; d]; d];
    for w in windows {
        for i in 0..d {
            let di = w[i] - mean[i];
            for j in i..d {
                cov[i][j] += di * (w[j] - mean[j]) / n;
            }
        }
    }
    for i in 0..d {
        for j in 0..i {
            cov[i][j] = cov[j][i];
        }
    }
    let (values, vectors) = linalg::sym_eigen(&cov);
    WindowModel { side, mean, comps: vectors[..k].to_vec(), variances: values[..k].to_vec() }
}

/// C4. Complete one window: find the coefficients `c` of the components that best explain the
/// visible pixels (`visible[i]` true), with a ridge that keeps each coefficient within its
/// variance: minimise `Σ_visible (mean + Σ_a c_a u_a − window)² + Σ_a lambda c_a² / variance_a`.
/// Returns the whole fitted window `mean + Σ_a c_a u_a` (row by row).
pub fn complete_window(m: &WindowModel, window: &[f64], visible: &[bool], lambda: f64) -> Vec<f64> {
    let k = m.comps.len();
    let mut c = vec![0.0; k];
    if k > 0 {
        let mut a = vec![vec![0.0; k]; k];
        let mut b = vec![0.0; k];
        for i in 0..window.len() {
            if !visible[i] {
                continue;
            }
            let r = window[i] - m.mean[i];
            for p in 0..k {
                b[p] += m.comps[p][i] * r;
                for q in 0..k {
                    a[p][q] += m.comps[p][i] * m.comps[q][i];
                }
            }
        }
        for p in 0..k {
            a[p][p] += lambda / m.variances[p];
        }
        c = linalg::solve(&a, &b).unwrap_or(c);
    }
    (0..window.len()).map(|i| m.mean[i] + (0..k).map(|p| c[p] * m.comps[p][i]).sum::<f64>()).collect()
}

/// C5. The PSNR (peak 1) over the hidden pixels only.
pub fn hidden_psnr(result: &Image<f32>, truth: &Image<f32>, mask: &Image<f32>) -> f32 {
    let (mut s, mut n) = (0.0f64, 0usize);
    for (x, y) in mask.coords() {
        if mask[(x, y)] > 0.5 {
            s += ((result[(x, y)] - truth[(x, y)]) as f64).powi(2);
            n += 1;
        }
    }
    (10.0 * (1.0 / (s / n as f64)).log10()) as f32
}
