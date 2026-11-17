// Coding exercises for "Box filters & local statistics". Replace each `todo!()` with your code,
// then check it with `cargo test -p integral box_filters::` and look at `cargo run -p integral -- box-filters`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

pub use crate::common::integral;

/// The sum over columns `x0..=x1`, rows `y0..=y1` from a table (given): four lookups.
pub fn rect_sum(s: &Image<f64>, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
    s[(x1 + 1, y1 + 1)] - s[(x0, y1 + 1)] - s[(x1 + 1, y0)] + s[(x0, y0)]
}

/// The window of radius `r` around `c` along an axis of length `n`, clipped to `0..n`:
/// returns the first and last index inside (given).
pub fn clip(c: usize, r: usize, n: usize) -> (usize, usize) {
    (c.saturating_sub(r), (c + r).min(n - 1))
}

/// C1. The box mean of width `2r + 1` along a row, with a running sum: start with the window
/// at position 0, then move one step at a time, adding the pixel that enters and subtracting
/// the one that leaves. At the ends the window is clipped; divide by the number of pixels
/// inside (exercise B1).
///
/// Hint: keep the window sum in a variable. Moving from position `i - 1` to `i`, pixel `i + r`
/// enters (if it exists) and pixel `i - r - 1` leaves (if it was inside). `clip(i, r, n)` gives
/// the first and last pixel of the window, so the count is `b - a + 1`.
pub fn box_mean_1d(f: &[f32], r: usize) -> Vec<f32> {
    todo!("C1: box_mean_1d")
}

/// C2. The box mean of width `2r + 1` at every pixel, from the integral image: clip the window
/// to the image and divide by the number of pixels inside (slide "Why the integral image wins
/// anyway").
///
/// Hint: build the table with `integral(img, |v| v)` once; for every pixel, `clip` the window
/// in x and in y, then `rect_sum`.
pub fn box_mean(img: &Image<f32>, r: usize) -> Image<f32> {
    todo!("C2: box_mean")
}

/// C3. The local mean `m` and local standard deviation `s` in the window of width `2r + 1`
/// (clipped) at every pixel, from two tables, of `I` and of `I²`: `s² = mean of I² − m²`
/// (slide "Local mean and local standard deviation"). A variance that comes out below zero
/// (rounding) counts as 0.
///
/// Hint: two tables, `integral(img, |v| v)` and `integral(img, |v| v * v)`; compute in `f64`
/// and convert with `as f32` at the end.
pub fn local_mean_sd(img: &Image<f32>, r: usize) -> (Image<f32>, Image<f32>) {
    todo!("C3: local_mean_sd")
}

/// C4. Local contrast normalisation `(I − m) / (s + eps)` with the window of width `2r + 1`
/// (slide "Local contrast normalisation").
///
/// Hint: `local_mean_sd`, then `map_xy`.
pub fn normalise(img: &Image<f32>, r: usize, eps: f32) -> Image<f32> {
    todo!("C4: normalise")
}

/// C5. `n` passes of the box mean of width `2r + 1` (slide "Repeated boxes look Gaussian").
///
/// Hint: start from `img.clone()` and apply `box_mean` `n` times.
pub fn repeated_box(img: &Image<f32>, r: usize, n: usize) -> Image<f32> {
    todo!("C5: repeated_box")
}
