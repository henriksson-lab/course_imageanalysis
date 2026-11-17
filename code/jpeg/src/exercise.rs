// Coding exercises for "The JPEG pipeline, artefacts and quality metrics". Replace each
// `todo!()` with your code, then check it with `cargo test -p jpeg` and look at `cargo run -p jpeg`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;
use std::f32::consts::PI;

/// The standard luminance quantisation table (quality 50), `table[row][column]`: row = vertical
/// frequency v, column = horizontal frequency u.
pub const LUMINANCE: [[u32; 8]; 8] = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
];

/// An AC symbol: `(run, size, value)` = the number of zeros before a non-zero value, the size
/// (number of bits) of the value, and the value. `(0, 0, 0)` is EOB (end of block) and
/// `(15, 0, 0)` is ZRL (a run of 16 zeros).
pub type Symbol = (u32, u32, i32);
pub const EOB: Symbol = (0, 0, 0);
pub const ZRL: Symbol = (15, 0, 0);

/// The 2D DCT of an 8 × 8 block (from the DCT unit): `c[(u, v)]`, u horizontal, v vertical.
pub fn dct8(block: &Image<f32>) -> Image<f32> {
    let a = |k: usize| if k == 0 { (1.0f32 / 8.0).sqrt() } else { 0.5 };
    Image::from_fn(8, 8, |u, v| {
        let mut s = 0.0;
        for (x, y) in block.coords() {
            s += block[(x, y)] * (PI * (2 * x + 1) as f32 * u as f32 / 16.0).cos() * (PI * (2 * y + 1) as f32 * v as f32 / 16.0).cos();
        }
        a(u) * a(v) * s
    })
}

/// The inverse of `dct8`.
pub fn idct8(c: &Image<f32>) -> Image<f32> {
    let a = |k: usize| if k == 0 { (1.0f32 / 8.0).sqrt() } else { 0.5 };
    Image::from_fn(8, 8, |x, y| {
        let mut s = 0.0;
        for (u, v) in c.coords() {
            s += a(u) * a(v) * c[(u, v)] * (PI * (2 * x + 1) as f32 * u as f32 / 16.0).cos() * (PI * (2 * y + 1) as f32 * v as f32 / 16.0).cos();
        }
        s
    })
}

/// C1. The quantisation table for a quality factor 1..=100 (slide "The math: the quality
/// factor"): s = 5000 / q for q < 50, s = 200 − 2q otherwise; every entry becomes
/// floor((Q s + 50) / 100), kept between 1 and 255.
///
/// Hint: integer arithmetic does the floor for you: `(LUMINANCE[v][u] * s + 50) / 100`, then
/// `.clamp(1, 255)`.
pub fn scale_table(quality: u32) -> [[u32; 8]; 8] {
    todo!("C1: scale_table")
}

/// C2a. Quantise the 64 coefficients of a block: q = round(c / Q).
///
/// Hint: `c.map_xy(|u, v, value| ...)` gives you the position of each coefficient.
pub fn quantise(c: &Image<f32>, table: &[[u32; 8]; 8]) -> Image<i32> {
    todo!("C2a: quantise")
}

/// C2b. Dequantise: c = q · Q.
pub fn dequantise(q: &Image<i32>, table: &[[u32; 8]; 8]) -> Image<f32> {
    todo!("C2b: dequantise")
}

/// C3. The 64 values of a block in zigzag order (slide "Zigzag and entropy coding"): along the
/// anti-diagonals u + v = 0, 1, ..., 14, alternating direction, starting (row, column) =
/// (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), ...
///
/// Hint: for the diagonal `s = row + column`, the rows run from `max(0, s − 7)` to `min(s, 7)`;
/// `q[(column, row)]` reads a value (x is the column).
pub fn zigzag(q: &Image<i32>) -> Vec<i32> {
    todo!("C3: zigzag")
}

/// C4a. The size of a value: the number of bits of |v| (0 → 0, 1 → 1, 2–3 → 2, 4–7 → 3, ...).
///
/// Hint: `v.unsigned_abs()`; count how often you can halve it before it reaches 0.
pub fn size_of(v: i32) -> u32 {
    todo!("C4a: size_of")
}

/// C4b. The AC symbols of a block from its zigzag sequence of 64 values (skip the DC, `zz[0]`):
/// for every non-zero value, (run of zeros before it, size, value). Runs longer than 15 are
/// split off as ZRL symbols. The block ends with EOB, unless its last value is not zero.
///
/// Hint: count zeros in `run`; at a non-zero value, first push a ZRL for every 16 zeros.
pub fn ac_symbols(zz: &[i32]) -> Vec<Symbol> {
    todo!("C4b: ac_symbols")
}

/// C5. The peak signal-to-noise ratio in dB: 10 log10(peak² / MSE).
pub fn psnr(a: &Image<f32>, b: &Image<f32>, peak: f32) -> f32 {
    todo!("C5: psnr")
}
