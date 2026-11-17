// Coding exercises for "Transform coding, the DCT and the KLT". Replace each `todo!()` with
// your code, then check it with `cargo test -p dct` and look at `cargo run -p dct`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;
use std::f32::consts::PI;

/// C1. The N × N DCT-II matrix, row by row: `d[k][n] = α_k cos(π (2n + 1) k / 2N)` with
/// α_0 = √(1/N) and α_k = √(2/N) for k ≥ 1 (slide "The math: the DCT-II"). Row k is the
/// k-th basis vector.
///
/// Hint: `let mut d = vec![vec![0.0; n]; n];` and two loops.
pub fn dct_matrix(n: usize) -> Vec<Vec<f32>> {
    todo!("C1: dct_matrix")
}

/// C2a. The DCT of a signal: c = D x (one coefficient per row of D).
pub fn dct(x: &[f32]) -> Vec<f32> {
    todo!("C2a: dct")
}

/// C2b. The inverse DCT: x = Dᵀ c, the signal as a sum of basis vectors weighted by the
/// coefficients.
///
/// Hint: the same matrix as `dct`, used the other way round: `x[n] += d[k][n] * c[k]`.
pub fn idct(c: &[f32]) -> Vec<f32> {
    todo!("C2b: idct")
}

/// C3a. The 2D DCT of a square block, C = D X Dᵀ: the 1D DCT of every row, then of every
/// column of the result. `out[(u, v)]` is the coefficient of horizontal frequency u and
/// vertical frequency v, so (0, 0) is the DC and the top row (v = 0) holds the horizontal
/// frequencies.
///
/// Hint: `dct(block.row(y))` transforms row `y`; collect a column into a `Vec` first.
pub fn dct2(block: &Image<f32>) -> Image<f32> {
    todo!("C3a: dct2")
}

/// C3b. The inverse 2D DCT: the inverse 1D DCT of every row, then of every column.
pub fn idct2(coeffs: &Image<f32>) -> Image<f32> {
    todo!("C3b: idct2")
}

/// C4. Block transform coding without quantisation: cut the image into 8 × 8 blocks, take the
/// DCT of every block, keep only the `fraction` of all coefficients (over all blocks) with the
/// largest size, set the rest to 0, and transform back. Returns the rebuilt image.
/// The image size must be a multiple of 8.
///
/// Hint: put all coefficients into one image of the same size (block by block), sort their
/// sizes `|c|` from large to small to find the threshold, then rebuild block by block.
/// `img.crop(bx, by, 8, 8)` cuts out a block.
pub fn keep_largest_blocks(img: &Image<f32>, fraction: f32) -> Image<f32> {
    todo!("C4: keep_largest_blocks")
}
