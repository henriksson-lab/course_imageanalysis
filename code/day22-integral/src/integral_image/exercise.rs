// Coding exercises for "Integral images". Replace each `todo!()` with your code,
// then check it with `cargo test -p day22 integral_image::` and look at `cargo run -p day22 -- integral-image`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// C1. The running total of a row: `s[0] = 0` and `s[i] = f[0] + ... + f[i-1]`, so `s` is one
/// longer than `f` (slide "The running total").
///
/// Hint: `let mut s = vec![0.0; f.len() + 1];` then one addition per entry.
pub fn running_total(f: &[f64]) -> Vec<f64> {
    todo!("C1: running_total")
}

/// C1. The sum of the pixels `a..=b` of the row (counted from 0), from its running total `s`:
/// one subtraction, whatever the length of the range.
///
/// Hint: everything up to pixel `b`, minus everything before pixel `a`.
pub fn range_sum(s: &[f64], a: usize, b: usize) -> f64 {
    todo!("C1: range_sum")
}

/// C2. The integral image of `img`, one larger than the image in each direction:
/// `table[(x, y)]` is the sum of the pixels in columns `0..x` and rows `0..y` (not including
/// `x` and `y`), so row 0 and column 0 are zero. Built in one pass with the recursion of the
/// slide "Building the table in one pass".
///
/// Hint: pixel `(x, y)` of the image is entry `(x + 1, y + 1)` of the table; its left,
/// upper and upper-left neighbours in the table are already filled when you go row by row.
/// Convert a pixel with `img[(x, y)] as f64`.
pub fn integral_image(img: &Image<f32>) -> Image<f64> {
    todo!("C2: integral_image")
}

/// C3. The sum of the pixels in columns `x0..=x1` and rows `y0..=y1` (counted from 0), from the
/// table: four lookups (slide "Any rectangle from four corners").
///
/// Hint: the bottom-right corner is entry `(x1 + 1, y1 + 1)`; the corner just outside the
/// rectangle, above and to the left, is entry `(x0, y0)`.
pub fn rect_sum(s: &Image<f64>, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
    todo!("C3: rect_sum")
}

/// C4. The same table in two passes: running totals along every row, then running totals of
/// the result down every column (exercise B8).
///
/// Hint: in the first pass, `s[(x + 1, y + 1)]` is the entry to its left plus one pixel.
pub fn integral_two_pass(img: &Image<f32>) -> Image<f64> {
    todo!("C4: integral_two_pass")
}

/// C5. The sum of every `k × k` window that fits inside the image, added up directly
/// (`k * k` pixels per window). The window with top-left corner `(x, y)` goes to `out[(x, y)]`.
///
/// Hint: the output is `(width + 1 - k) × (height + 1 - k)`.
pub fn window_sums_direct(img: &Image<f32>, k: usize) -> Image<f64> {
    todo!("C5: window_sums_direct")
}

/// C5. The same window sums from the table: four lookups per window.
///
/// Hint: build the table once, then call `rect_sum` for every window.
pub fn window_sums_table(img: &Image<f32>, k: usize) -> Image<f64> {
    todo!("C5: window_sums_table")
}
