// Coding exercises for "Edge-stopping diffusion (Perona–Malik)". Replace each `todo!()` with your
// code, then check it with `cargo test -p day10 perona_malik::` and look at `cargo run -p day10 -- perona-malik`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};

/// C1a. The edge-stopping function `g1(s) = exp(-(s/K)^2)`.
///
/// Hint: `(s / k).powi(2)` is `(s/K)^2`, and `x.exp()` is `e^x`.
pub fn g1(s: f32, k: f32) -> f32 {
    todo!("C1a: g1")
}

/// C1b. The edge-stopping function `g2(s) = 1 / (1 + (s/K)^2)`.
pub fn g2(s: f32, k: f32) -> f32 {
    todo!("C1b: g2")
}

/// An edge-stopping function: `g(size of the difference, K)`, for example `g1` or `g2`.
pub type Stop = fn(f32, f32) -> f32;

/// C2. One Perona–Malik step on a row (slide "One step by hand"): between every pair of
/// neighbours the flow is `g(|d|) * d`, `d` = the difference; every value gets `dt` times
/// the flows coming in. Mirror border: nothing flows out at the two ends.
///
/// Hint: first compute the flow of every pair `(i, i+1)`, `g(d.abs(), k) * d` with
/// `d = row[i+1] - row[i]`; then value `i` gains `dt` times the flow from its right pair and
/// loses `dt` times the flow of its left pair. A function passed as `g` is called as `g(s, k)`.
pub fn pm_step_1d(row: &[f32], dt: f32, k: f32, g: Stop) -> Vec<f32> {
    todo!("C2: pm_step_1d")
}

/// C3. One Perona–Malik step on an image: the four flows of the heat step, each multiplied
/// by `g(|difference|)`. With `sigma > 0` the conductivity is computed from the differences
/// of the image blurred with a Gaussian of that σ (slide "Price 2: ill-posed, and the cure"),
/// while the flows still use the differences of the image itself.
///
/// Hint: as the heat step, but each of the four differences `d` is multiplied by `g(|ds|, k)`,
/// where `ds` is the same difference in the blurred image (`gaussian_blur`, below), or in `img`
/// itself when `sigma` is 0.
pub fn pm_step(img: &Image<f32>, dt: f32, k: f32, g: Stop, sigma: f32) -> Image<f32> {
    todo!("C3: pm_step")
}

/// C4. The peak signal-to-noise ratio of `result` against the true image `truth`, in dB,
/// for grey values with a peak of `peak` (1 for the course's images, 255 for 8-bit numbers):
/// `10 log10(peak^2 / mean squared error)`.
pub fn psnr(result: &Image<f32>, truth: &Image<f32>, peak: f32) -> f32 {
    todo!("C4: psnr")
}

/// Not an exercise: a Gaussian blur (radius 4σ, mirror border), in `src/common.rs`.
pub use crate::common::gaussian_blur;
