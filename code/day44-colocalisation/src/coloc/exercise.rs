// Coding exercises for "Colocalisation of two channels". Replace each `todo!()` with your
// code, then check it with `cargo test -p day44 coloc::` and look at `cargo run -p day44 -- coloc`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

// All functions take the two channels `a`, `b` and a `mask`: only pixels where the mask is
// `true` (the region of interest) count.

/// C1. Pearson's correlation of the two channels over the pixels of the mask
/// (slide "The math: correlation and overlap").
///
/// Hint: two passes over the pixels: first the means, then the three sums of products of
/// the differences from the means. Sums in `f64` are safer for large images.
pub fn pearson(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>) -> f32 {
    todo!("C1: pearson")
}

/// C2. The overlap coefficient and Manders' k1, k2: with the uncentred sums
/// S_AB = Σ A·B, S_AA = Σ A², S_BB = Σ B², returns
/// `(S_AB / sqrt(S_AA · S_BB), S_AB / S_AA, S_AB / S_BB)`.
pub fn overlap(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>) -> (f32, f32, f32) {
    todo!("C2: overlap")
}

/// C3. Manders' M1 and M2 with thresholds `ta`, `tb` ("above" means strictly greater):
/// M1 = Σ A over pixels above in both / Σ A over pixels above in A, and M2 the same for B.
pub fn manders(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>, ta: f32, tb: f32) -> (f32, f32) {
    todo!("C3: manders")
}

/// C4a. The ranks of a list of values: 1 for the smallest, n for the largest; values that
/// are equal share the average of their ranks (two values tied for ranks 1 and 2 both get 1.5).
///
/// Hint: sort the positions `0..n` by their value
/// (`order.sort_by(|&i, &j| values[i].total_cmp(&values[j]))`), then walk through the
/// sorted list in groups of equal values.
pub fn ranks(values: &[f32]) -> Vec<f32> {
    todo!("C4a: ranks")
}

/// C4b. Spearman's coefficient: Pearson's r of the ranks of the pixels in the mask
/// (slide "Rank and robustness").
///
/// Hint: collect the masked values of each channel (in the order of `a.coords()`), rank
/// them with `ranks`, put the ranks back into two images, and call `pearson`.
pub fn spearman(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>) -> f32 {
    todo!("C4b: spearman")
}

/// C5a. The orthogonal regression line B = slope · A + intercept through the scatter plot
/// (slide "The math: the regression line"): with the centred sums of squares and products
/// s_AA, s_BB, s_AB, slope = (s_BB − s_AA + sqrt((s_BB − s_AA)² + 4 s_AB²)) / (2 s_AB).
/// Returns `(slope, intercept)`.
pub fn orthogonal_line(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>) -> (f32, f32) {
    todo!("C5a: orthogonal_line")
}

/// C5b. Costes' thresholds (slide "Thresholds: Costes' method"): start t_A at the largest A
/// in the mask and lower it in `steps` equal steps down to the smallest; t_B = slope · t_A +
/// intercept follows the line. Stop at the first step where Pearson's r of the pixels
/// **below** the thresholds (A ≤ t_A or B ≤ t_B, inside the mask) is ≤ 0, and return
/// `Some((t_A, t_B))` there (or at the last step). `None` if r of the whole mask is ≤ 0: then
/// there is no rising line to walk down.
///
/// Hint: for each step build the mask of the pixels below with `Image::from_fn` and call
/// `pearson`. A mask with too few pixels gives r = NaN; `NaN <= 0.0` is false, so the
/// search simply goes on.
pub fn costes(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>, steps: usize) -> Option<(f32, f32)> {
    todo!("C5b: costes")
}
