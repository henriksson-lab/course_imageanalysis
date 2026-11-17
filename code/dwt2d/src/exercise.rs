// Coding exercises for "The 2D wavelet transform and sparsity of images". Replace each
// `todo!()` with your code, then check it with `cargo test -p dwt2d` and look at `cargo run -p dwt2d`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

// ---- From the filter bank unit (given): orthogonal low-pass filters and one 1D level. ----

pub const HAAR: [f32; 2] = [std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2];
pub const DB2: [f32; 4] = [0.482_962_9, 0.836_516_3, 0.224_143_87, -0.129_409_52];
pub const DB4: [f32; 8] = [0.230_377_81, 0.714_846_6, 0.630_880_8, -0.027_983_77, -0.187_034_81, 0.030_841_38, 0.032_883_01, -0.010_597_4];

/// The high-pass filter `g[n] = (−1)^n h[L−1−n]`.
pub fn highpass(h: &[f32]) -> Vec<f32> {
    let l = h.len();
    (0..l).map(|n| if n % 2 == 0 { h[l - 1 - n] } else { -h[l - 1 - n] }).collect()
}

/// One level of the 1D analysis filter bank, periodic: returns `a` followed by `d` in one vector.
pub fn analysis_1d(x: &[f32], h: &[f32]) -> Vec<f32> {
    let g = highpass(h);
    let n = x.len();
    let mut out = vec![0.0; n];
    for k in 0..n / 2 {
        for i in 0..h.len() {
            out[k] += h[i] * x[(2 * k + i) % n];
            out[n / 2 + k] += g[i] * x[(2 * k + i) % n];
        }
    }
    out
}

/// One level of the 1D synthesis filter bank, periodic: the inverse of `analysis_1d`.
pub fn synthesis_1d(c: &[f32], h: &[f32]) -> Vec<f32> {
    let g = highpass(h);
    let n = c.len();
    let mut x = vec![0.0; n];
    for k in 0..n / 2 {
        for i in 0..h.len() {
            x[(2 * k + i) % n] += h[i] * c[k] + g[i] * c[n / 2 + k];
        }
    }
    x
}

// ---- The exercises. ----

/// C1. One level of the separable 2D transform: `analysis_1d` on every row, then on every
/// column of the result (slide "Separable: rows, then columns"). The four quarter-size bands
/// sit in place: LL top left, V (high-pass along x) top right, H (high-pass along y) bottom
/// left, D bottom right. Width and height must be even.
///
/// Hint: a new image `rows` with `analysis_1d` of every `img.row(y)`; then `analysis_1d` of
/// every column of `rows` (collect it into a `Vec<f32>`), written into `out`.
pub fn dwt2d_level(img: &Image<f32>, h: &[f32]) -> Image<f32> {
    todo!("C1: dwt2d_level")
}

/// C2. The inverse of C1: `synthesis_1d` on every column, then on every row.
///
/// Hint: undo C1 in reverse order: columns first, then rows.
pub fn idwt2d_level(c: &Image<f32>, h: &[f32]) -> Image<f32> {
    todo!("C2: idwt2d_level")
}

/// C3a. The subband pyramid (Mallat layout): `levels` times C1, each time on the LL band in the
/// top-left corner, which halves in width and height every level (slide "The subband pyramid").
///
/// Hint: keep the size `(w, h)` of the current LL band; transform `c.crop(0, 0, w, h)` with C1,
/// copy the result back into the top-left corner of `c`, and halve `w` and `h`.
pub fn dwt2d(img: &Image<f32>, h: &[f32], levels: usize) -> Image<f32> {
    todo!("C3a: dwt2d")
}

/// C3b. The inverse of C3a: C2 on the top-left corner, from the coarsest level out.
///
/// Hint: the coarsest LL band is `width >> levels` by `height >> levels`; double the size
/// before each step.
pub fn idwt2d(c: &Image<f32>, h: &[f32], levels: usize) -> Image<f32> {
    todo!("C3b: idwt2d")
}

/// C4a. Keep the given fraction (0 to 1) of all coefficients, the largest in magnitude; set the
/// others to 0. Returns the kept coefficients and the sum of the squares of the dropped ones.
///
/// Hint: sort the magnitudes, largest first; the `k`-th largest (`k = fraction · number of
/// coefficients`, rounded) is the threshold. Keep every coefficient at least that large.
pub fn keep_fraction(c: &Image<f32>, fraction: f32) -> (Image<f32>, f32) {
    todo!("C4a: keep_fraction")
}

/// C4b. The peak signal-to-noise ratio in dB: `10 log10(peak² / MSE)`, with the mean squared
/// difference between the two images (slide "The math: K-term approximation and PSNR").
pub fn psnr(a: &Image<f32>, b: &Image<f32>, peak: f32) -> f32 {
    todo!("C4b: psnr")
}
