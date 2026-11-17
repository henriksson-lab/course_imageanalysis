// Coding exercises for "Bleed-through and compensation". Replace each `todo!()` with your
// code, then check it with `cargo test -p unmixing crosstalk::` and look at `cargo run -p unmixing -- crosstalk`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// A 2 × 2 matrix, row by row: `m[row][column]`; rows = channels, columns = dyes.
pub type Mat2 = [[f32; 2]; 2];

/// C1a. The value a channel records for one unit of a dye: the dye's emission spectrum times
/// the filter's transmission, added over the wavelength bins, times the bin width `dl`.
pub fn channel_value(spectrum: &[f32], filter: &[f32], dl: f32) -> f32 {
    todo!("C1a: channel_value")
}

/// C1b. The mixing matrix of two dyes and two channels: entry `[c][k]` is what channel c
/// records for one unit of dye k. Column k is dye k as the channels see it.
pub fn mixing_matrix(dyes: [&[f32]; 2], filters: [&[f32]; 2], dl: f32) -> Mat2 {
    todo!("C1b: mixing_matrix")
}

/// C2. The slope of the line through the origin that fits the points (x_i, y_i) best in the
/// least-squares sense: `Σ x y / Σ x²` (a single-stained control gives a spillover).
pub fn slope_through_origin(xs: &[f32], ys: &[f32]) -> f32 {
    todo!("C2: slope_through_origin")
}

/// C3a. The inverse of a 2 × 2 matrix by the formula of the slide "The math: the 2 × 2 inverse".
pub fn inverse2(m: Mat2) -> Mat2 {
    todo!("C3a: inverse2")
}

/// C3b. Compensate a two-channel image: at every pixel, the dye amounts `a = S⁻¹ y` from the
/// channel values `y = (ch1, ch2)`. Returns the two dye images.
///
/// Hint: compute `inverse2(s)` once, outside the loop over the pixels.
pub fn compensate(ch1: &Image<f32>, ch2: &Image<f32>, s: Mat2) -> (Image<f32>, Image<f32>) {
    todo!("C3b: compensate")
}

/// C4. Pearson's correlation of two images over the pixels where `mask` is true:
/// `Σ (a − ā)(b − b̄) / sqrt(Σ (a − ā)² · Σ (b − b̄)²)`, with the means over the mask.
pub fn pearson(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>) -> f32 {
    todo!("C4: pearson")
}
