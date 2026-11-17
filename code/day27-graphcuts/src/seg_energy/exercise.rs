// Coding exercises for "Segmentation as energy minimisation". Replace each `todo!()`
// with your code, then check it with `cargo test -p day27 seg_energy::` and look at `cargo run -p day27 -- seg-energy`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// C1a. The data costs of the two labels at every pixel, from two Gaussians of equal width:
/// D(ℓ) = (I − μℓ)² / (2 s²) (slide "The data term from an intensity model"; the constant
/// is left out). Returns the images (D(0), D(1)).
///
/// Hint: `img.map(|v| ...)` applies the same formula to every pixel.
pub fn data_costs(img: &Image<f32>, mu0: f32, mu1: f32, s: f32) -> (Image<f32>, Image<f32>) {
    todo!("C1a: data_costs")
}

/// C1b. The energy of a labelling (labels 0 and 1): the data cost of every pixel's label,
/// plus `lambda` times the number of 4-neighbour pairs with different labels (w = 1).
///
/// Hint: to count every neighbour pair once, compare each pixel only with its right
/// neighbour and the one below (when they exist).
pub fn energy(labels: &Image<u32>, d0: &Image<f32>, d1: &Image<f32>, lambda: f32) -> f32 {
    todo!("C1b: energy")
}

/// C2. Thresholding as the minimum for λ = 0: every pixel takes its cheaper label
/// (1 where D(1) < D(0)).
///
/// Hint: `d0.zip_map(d1, |a, b| ...)` combines the two images pixel by pixel.
pub fn threshold(d0: &Image<f32>, d1: &Image<f32>) -> Image<u32> {
    todo!("C2: threshold")
}

/// C3. The exact minimum of the chain energy Σ D_k(L_k) + λ Σ [L_k ≠ L_k+1] by dynamic
/// programming (math slide "Dynamic programming on a chain"). Returns the best labelling
/// and its energy.
///
/// Hint: keep a table `c` of n rows `[f32; 2]` (best energy so far with this pixel 0 or 1)
/// and a table `from` of the same size (which label of the previous pixel gave it). Fill both
/// from left to right; then start at the better end and walk back through `from`.
pub fn chain_dp(d0: &[f32], d1: &[f32], lambda: f32) -> (Vec<u32>, f32) {
    todo!("C3: chain_dp")
}

/// C4. Dynamic programming on every row of an image separately: the exact minimum for each
/// row on its own, ignoring the boundaries between rows.
///
/// Hint: `d0.row(y)` is row y as a slice, ready for `chain_dp`.
pub fn rows_dp(d0: &Image<f32>, d1: &Image<f32>, lambda: f32) -> Image<u32> {
    todo!("C4: rows_dp")
}
