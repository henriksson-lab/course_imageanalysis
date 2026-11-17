// Coding exercises for "Compression and measurements, JPEG 2000, formats". Replace each
// `todo!()` with your code, then check it with `cargo test -p compression compress_science::` and look at
// `cargo run -p compression -- compress-science`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// Floor of a / b for whole numbers (b > 0), also for negative a: `floor_div(-3, 2) = -2`.
pub fn floor_div(a: i32, b: i32) -> i32 {
    a.div_euclid(b)
}

/// C1. One level of the reversible 5/3 wavelet by lifting (slide "The math: an exactly
/// reversible wavelet"), for a signal of even length. Returns (s, d): the smooth (low-pass)
/// and detail (high-pass) halves.
///
///   d[n] = x[2n+1] − floor((x[2n] + x[2n+2]) / 2)
///   s[n] = x[2n]   + floor((d[n−1] + d[n] + 2) / 4)
///
/// At the ends the signal is mirrored: x[len] means x[len − 2], and d[−1] means d[0].
///
/// Hint: compute all of `d` first, then `s` (it needs `d`). Use `floor_div` for the floors.
pub fn lift53(x: &[i32]) -> (Vec<i32>, Vec<i32>) {
    todo!("C1: lift53")
}

/// C2. The inverse of `lift53`: undo the update, then undo the prediction, with exactly the
/// same rounded terms. Integers in, the same integers out.
///
/// Hint: the even samples come first (`x[2n] = s[n] − ...`), then the odd ones, which need
/// their even neighbours.
pub fn unlift53(s: &[i32], d: &[i32]) -> Vec<i32> {
    todo!("C2: unlift53")
}

/// C3. One level of the 2D wavelet: `lift53` on every row, then on every column. The result has
/// the same size, with the four subbands in the four quarters: LL (smooth in both directions)
/// top left, then the detail bands. Width and height must be even.
///
/// Hint: `lift53(img.row(y))` for a row; put `s` in the left half and `d` in the right half.
/// Then the same for every column, `s` in the top half.
pub fn wavelet2d(img: &Image<i32>) -> Image<i32> {
    todo!("C3: wavelet2d")
}

/// C4. Bit planes: a decoder has received the top `planes` bits of a `bits`-bit value. Which
/// value does it guess? The middle of all values that start with those bits.
///
/// Hint: `(value >> k) << k` sets the lowest k bits to 0.
pub fn bitplane_guess(value: u32, bits: u32, planes: u32) -> f32 {
    todo!("C4: bitplane_guess")
}

/// C5. How many chunks a region read touches (slide "The math: how many chunks?"): the region
/// is `[a_d, b_d)` along each dimension d, the chunks have size `c_d` and start at 0.
pub fn chunks_read(region: &[(usize, usize)], chunk: &[usize]) -> usize {
    todo!("C5: chunks_read")
}
