// Coding exercises for "Back-projection and the Fourier slice theorem". Replace each `todo!()`
// with your code, then check it with `cargo test -p day09 backprojection::` and look at
// `cargo run -p day09 -- backprojection`.
// Coordinates as in the lecture: origin at the image centre, y up, θ counter-clockwise
// (see `src/common.rs`: `centre`, `projection`, `sinogram`, `angles`).
#![allow(unused_variables, unused_imports)]

use crate::common::centre;
use imgcourse::{Complex, Image};
use std::f32::consts::PI;

/// C1. Back-projection of two projections by hand (slide "By hand: three points, two angles"):
/// every pixel gets its column's value from `p0` plus its row's value from `p90`.
/// Bin k of `p0` is column k (x grows to the right); bin k of `p90` is the row at height
/// y = k − c, so bin 0 is the **bottom** row.
///
/// Hint: `Image::from_fn(n, n, |px, py| ...)`.
pub fn backproject_two(p0: &[f32], p90: &[f32]) -> Image<f32> {
    todo!("C1: backproject_two")
}

/// C2. Back-projection of a sinogram (`sino[(a, k)]` = bin k at `thetas[a]`) onto an n × n
/// image: for every pixel (x, y) and every angle, read the sinogram at ρ = x cos θ + y sin θ
/// (between bins: linear interpolation, 0 outside the detector); add over the angles and
/// multiply by π / N_θ (the step dθ of the integral).
///
/// Hint: pixel `(px, py)` is at x = px − c, y = c − py. The bin position is `ρ + cb` with
/// `cb = centre(sino.height())`; take `floor` and the fraction for the interpolation.
pub fn backproject(sino: &Image<f32>, thetas: &[f32], n: usize) -> Image<f32> {
    todo!("C2: backproject")
}

/// C3a. The size of the 1D DFT of a projection, in centred order: element i is frequency
/// ω = i − n/2, so ω = 0 is in the middle (element n/2).
///
/// Hint: `imgcourse::fft::fft(&values, false)` with `values` the projection as `Complex`
/// numbers (`Complex::from(v)`); frequency ω is at index `ω mod n`, so element i is
/// `spectrum[(i + n - n / 2) % n].abs()`.
pub fn projection_spectrum(p: &[f32]) -> Vec<f32> {
    todo!("C3a: projection_spectrum")
}

/// C3b. The size of the 2D DFT of an n × n image along the line through frequency 0 at angle θ,
/// in the same centred order: element i is the point ω (cos θ, sin θ) with ω = i − n/2, read
/// between frequencies by bilinear interpolation. In the DFT, index (u, v) is (x frequency,
/// row frequency), and rows go **down**: the point ω (cos θ, sin θ) is u = ω cos θ,
/// v = −ω sin θ. Frequencies wrap around: index −1 is n − 1.
///
/// Hint: `(u.rem_euclid(n), v.rem_euclid(n))` wraps a frequency (as `isize`) into the image.
pub fn spectrum_slice(spectrum: &Image<Complex>, theta: f32) -> Vec<f32> {
    todo!("C3b: spectrum_slice")
}
