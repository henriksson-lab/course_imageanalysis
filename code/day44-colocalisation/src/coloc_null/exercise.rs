// Coding exercises for "Is it more than chance?". Replace each `todo!()` with your code,
// then check it with `cargo test -p day44 coloc_null::` and look at `cargo run -p day44 -- coloc-null`.
#![allow(unused_variables, unused_imports)]

use crate::common::pearson;
use imgcourse::{Border, Image, Rng};

/// C1. The shift curve (slide "The shift curve"): for every shift dx = −max_shift, …, max_shift,
/// Pearson's r over the mask between A(x, y) and B(x + dx, y). Outside the image, B is read
/// with the nearest edge pixel. Returns the 2 · max_shift + 1 values, the shift 0 in the middle.
///
/// Hint: make the shifted B with `Image::from_fn` and `b.get_border(x + dx, y, Border::Clamp)`
/// (as `isize`), then call `pearson(a, &shifted, mask)` (given, from the last unit).
pub fn shift_curve(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>, max_shift: usize) -> Vec<f32> {
    todo!("C1: shift_curve")
}

/// C2. The sub-pixel position of a peak: the vertex of the parabola through the three values
/// at −1, 0 and +1 (slide "The math: the width of the peak"). Returns an offset from 0,
/// between −0.5 and 0.5 when the middle value is the largest.
pub fn subpixel_peak(r_minus: f32, r0: f32, r_plus: f32) -> f32 {
    todo!("C2: subpixel_peak")
}

/// C3. Block randomisation (slide "Randomisation"): cut the image into `block × block`
/// squares (a grid from the top left; squares that do not fit at the right or bottom edge stay
/// where they are), take the squares whose **centre** pixel is in the mask, and put their
/// contents back in a random order (every order equally likely). All other pixels stay.
///
/// Hint: collect the top-left corners of the squares inside, make a list `0..n` and shuffle
/// it (Fisher–Yates: for i from n − 1 down to 1, swap item i with item `rng.below(i + 1)`).
/// Then copy square `order[k]` of `b` to the place of square `k` in a copy of `b`.
pub fn block_shuffle(b: &Image<f32>, mask: &Image<bool>, block: usize, rng: &mut Rng) -> Image<f32> {
    todo!("C3: block_shuffle")
}

/// C4. The permutation p-value (slide "The math: the permutation p-value"):
/// (1 + number of null values ≥ r_obs) / (N + 1).
pub fn permutation_p(r_obs: f32, null: &[f32]) -> f32 {
    todo!("C4: permutation_p")
}

/// C5a. The product of the differences from the mean (PDM, slide "Li's ICQ"):
/// (A − mean A)(B − mean B) at every pixel of the mask, with the means over the mask;
/// 0 outside the mask.
pub fn pdm(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>) -> Image<f32> {
    todo!("C5a: pdm")
}

/// C5b. Li's ICQ: the share of the mask's pixels with a positive PDM, minus 0.5.
pub fn icq(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>) -> f32 {
    todo!("C5b: icq")
}
