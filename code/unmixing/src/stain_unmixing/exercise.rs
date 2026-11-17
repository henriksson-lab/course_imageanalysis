// Coding exercises for "Stain separation: colour deconvolution". Replace each `todo!()` with
// your code, then check it with `cargo test -p unmixing stain_unmixing::` and look at
// `cargo run -p unmixing -- stain-unmixing`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{linalg, Image};

/// A colour image of **linear** intensities: `[red, green, blue]` per pixel, in [0, 1].
pub type Rgb = Image<[f32; 3]>;
/// A 3 × 3 matrix, row by row: `m[row][column]`. A stain matrix has the stains' unit OD
/// vectors (red, green, blue) as its columns.
pub type Mat3 = [[f32; 3]; 3];

/// Published stain vectors (Ruifrok & Johnston), (red, green, blue), not yet of length 1.
pub const HEMATOXYLIN: [f32; 3] = [0.65, 0.70, 0.29];
pub const EOSIN: [f32; 3] = [0.07, 0.99, 0.11];
pub const DAB: [f32; 3] = [0.27, 0.57, 0.78];

/// C1. The optical density of every pixel, `OD_c = −log10(I_c / I0_c)` per channel. An
/// intensity below 1/255 is first raised to 1/255 (a black pixel has no finite OD).
pub fn optical_density(img: &Rgb, i0: [f32; 3]) -> Rgb {
    todo!("C1: optical_density")
}

/// C2. Beer–Lambert: the transmitted intensities of a pixel with stain amounts `amounts`
/// (one per column of `s`): `I_c = I0_c · 10^(−Σ_k s[c][k] · amounts[k])`.
pub fn transmit(amounts: [f32; 3], s: &Mat3, i0: [f32; 3]) -> [f32; 3] {
    todo!("C2: transmit")
}

/// C3a. The stain matrix of two stains (slide "The stain matrix and colour deconvolution"):
/// columns = the two vectors scaled to length 1, and their cross product, scaled to length 1
/// (the residual direction).
pub fn stain_matrix(s1: [f32; 3], s2: [f32; 3]) -> Mat3 {
    todo!("C3a: stain_matrix")
}

/// C3b. Colour deconvolution: the three concentration images `C = S⁻¹ OD`, pixel by pixel.
///
/// Hint: column j of S⁻¹ solves `S x = e_j`: three calls of `linalg::solve` (in f64).
/// Compute S⁻¹ once, then apply it to every pixel.
pub fn deconvolve(od: &Rgb, s: &Mat3) -> [Image<f32>; 3] {
    todo!("C3b: deconvolve")
}

/// C4. Macenko's estimate of two stain vectors from the OD of the pixels
/// (slide "The math: Macenko in formulas"):
/// 1. keep the pixels with |od| > `beta`;
/// 2. v1, v2 = the two leading eigenvectors of Σ od odᵀ (no centring);
/// 3. every kept pixel's angle φ = atan2(od · v2, od · v1);
/// 4. the stain vectors are cos φ v1 + sin φ v2 at the `alpha` and the `100 − alpha`
///    percentiles of the angles.
///
/// Returns the two unit vectors, the one with the larger red OD (hematoxylin) first. (An
/// eigenvector may come out pointing either way: flip v1 if its entries sum to less than 0,
/// so that it points into the cloud of positive ODs.)
///
/// Hint: `linalg::sym_eigen` returns the eigenvectors with the largest eigenvalue first.
/// Sort the angles with `angles.sort_by(f32::total_cmp)` and take the entry at
/// `(p / 100 · (n − 1)).round()`.
pub fn macenko(od: &[[f32; 3]], beta: f32, alpha: f32) -> [[f32; 3]; 2] {
    todo!("C4: macenko")
}
