// Coding exercises for "The 2D wavelet transform and sparsity of images". Replace each
// `todo!()` with your code, then check it with `cargo test -p day07 dwt2d::` and look at `cargo run -p day07 -- dwt2d`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

// ---- From the filter bank unit (given, in `src/common.rs`): orthogonal low-pass filters and
// one 1D level. ----

pub use crate::common::{analysis_1d, highpass, synthesis_1d, DB2, DB4, HAAR};

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
