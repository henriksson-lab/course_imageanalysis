// Coding exercises for "Acceptor photobleaching". Replace each `todo!()` with your code,
// then check it with `cargo test -p fret acceptor_bleach::` and look at `cargo run -p fret -- acceptor-bleach`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};

/// C1a. The efficiency at every pixel (slide "How much brighter?"): E = 1 − pre / post;
/// 0 where `post` is not positive.
pub fn efficiency_map(pre: &Image<f32>, post: &Image<f32>) -> Image<f32> {
    todo!("C1a: efficiency_map")
}

/// C1b. The efficiency of a region as a ratio of sums (slide "Pixel-wise or region-wise?"):
/// E = 1 − Σ pre / Σ post over the mask.
pub fn region_efficiency(pre: &Image<f32>, post: &Image<f32>, mask: &Image<bool>) -> f32 {
    todo!("C1b: region_efficiency")
}

/// C2a. The correction for incomplete bleaching (slide "Incomplete bleaching"): with a
/// fraction `b` of the acceptor bleached, E = E_obs / (b + (1 − b) E_obs).
pub fn correct_incomplete(e_obs: f32, b: f32) -> f32 {
    todo!("C2a: correct_incomplete")
}

/// C2b. The control ratio for donor bleaching (slide "Donor bleaching, motion,
/// photoconversion"): Σ post / Σ pre over a region that was not bleached. Dividing the post
/// image by it undoes the donor's own bleaching.
pub fn control_ratio(pre: &Image<f32>, post: &Image<f32>, control: &Image<bool>) -> f32 {
    todo!("C2b: control_ratio")
}

/// C3a. A whole-pixel shift: the result at (x, y) is `img` at (x + dx, y + dy), 0 outside.
pub fn shift(img: &Image<f32>, dx: isize, dy: isize) -> Image<f32> {
    todo!("C3a: shift")
}

/// C3b. How far did the cells move between `before` and `after`? The whole-pixel shift
/// (dx, dy), each from −max_shift to max_shift, for which `shift(after, dx, dy)` best matches
/// `before`: the largest Σ before(x, y) · after(x + dx, y + dy). Then
/// `shift(after, dx, dy)` is aligned with `before`.
///
/// Hint: this is the "slide and score" of the registration day; a plain double loop over the
/// shifts with a sum over the pixels is fast enough for a few pixels of drift.
pub fn find_shift(before: &Image<f32>, after: &Image<f32>, max_shift: isize) -> (isize, isize) {
    todo!("C3b: find_shift")
}

/// C4. The noise of E = 1 − P/Q for photon counts P (before) and Q (after)
/// (slide "The math: the noise of E = 1 − P/Q"): σ_E ≈ (P/Q) sqrt(1/P + 1/Q).
pub fn sigma_e(p: f32, q: f32) -> f32 {
    todo!("C4: sigma_e")
}
