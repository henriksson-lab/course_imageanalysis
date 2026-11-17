// Coding exercises for "Denoising, artefacts, and what you may measure". Replace each `todo!()`
// with your code, then check it with `cargo test -p day05 denoise::` and look at `cargo run -p day05 -- denoise`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};

/// C1. The median filter: every output pixel is the middle value of the
/// `(2 radius + 1) × (2 radius + 1)` window around it (mirror border).
///
/// Hint: collect the window's values in a `Vec`, sort it with `sort_by(f32::total_cmp)`, and
/// take the element in the middle.
pub fn median_filter(img: &Image<f32>, radius: usize) -> Image<f32> {
    todo!("C1: median_filter")
}

/// C2. The bilateral filter: a weighted mean over the `(2 radius + 1)²` window (mirror border),
/// each neighbour `q` of the pixel `p` weighted by closeness and similarity:
/// `w = exp(−|p − q|² / 2σs²) · exp(−(I(p) − I(q))² / 2σr²)`.
///
/// Hint: keep two sums, `sum of w · I(q)` and `sum of w`; the result is their ratio.
pub fn bilateral(img: &Image<f32>, radius: usize, sigma_s: f32, sigma_r: f32) -> Image<f32> {
    todo!("C2: bilateral")
}

/// C3. Non-local means: a weighted mean over the pixels `q` of the search window
/// (`(2 search + 1)²` around `p`, only inside the image). The weight compares the patches
/// (`(2 patch + 1)²`, mirror border) around `p` and `q`: `w = exp(−d² / h²)`, with `d²` the
/// mean squared difference of the two patches.
///
/// Hint: four loops: over the search window (`qx`, `qy`; skip positions outside the image with
/// `img.contains`), and inside, over the patch offsets (`dx`, `dy`) to sum the squared
/// differences.
pub fn nonlocal_means(img: &Image<f32>, patch: usize, search: usize, h: f32) -> Image<f32> {
    todo!("C3: nonlocal_means")
}
