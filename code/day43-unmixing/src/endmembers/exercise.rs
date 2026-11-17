// Coding exercises for "Endmembers: the geometry of mixtures". Replace each `todo!()` with
// your code, then check it with `cargo test -p day43 endmembers::` and look at `cargo run -p day43 -- endmembers`.
#![allow(unused_variables, unused_imports)]

use imgcourse::linalg;

/// A matrix, row by row: `m[row][column]`. Spectra are the columns of M (rows = channels);
/// in Y the pixels are the columns (rows = channels, one column per pixel).
pub type Matrix = Vec<Vec<f32>>;

/// R1. Non-negative least squares for a few components: the abundances a ≥ 0 that minimise
/// ‖y − M a‖². Try every subset of the components ("which ones are not zero"): solve ordinary
/// least squares with only those columns; keep the solutions whose abundances are all ≥ 0;
/// return the one with the smallest residual (the others are 0). The empty subset (all zero)
/// is allowed too.
///
/// Hint: a subset of K components is a number `mask` in `0..(1 << K)`: component k is in it
/// when `mask & (1 << k) != 0`. `linalg::least_squares(&rows, &y)` solves least squares
/// (in f64, `rows` row by row).
pub fn nnls(m: &Matrix, y: &[f32]) -> Vec<f32> {
    todo!("R1: nnls")
}

/// R2. The fractions (barycentric coordinates) of the point `p` in the triangle with corners
/// `e[0]`, `e[1]`, `e[2]`: `p = a0 e0 + a1 e1 + a2 e2` with a0 + a1 + a2 = 1. A negative
/// fraction means the point is outside the triangle.
///
/// Hint: with a0 = 1 − a1 − a2 this is `a1 (e1 − e0) + a2 (e2 − e0) = p − e0`: a 2 × 2 system.
pub fn fractions(p: [f32; 2], e: [[f32; 2]; 3]) -> [f32; 3] {
    todo!("R2: fractions")
}

/// R3. The singular values of the pixel matrix Y (channels × pixels), largest first: the
/// square roots of the eigenvalues of `Y Yᵀ` (channels × channels, no centring).
///
/// Hint: `linalg::sym_eigen(&yyt)` returns the eigenvalues largest first (in f64); a tiny
/// negative value from rounding should become 0 before the square root.
pub fn singular_values(y: &Matrix) -> Vec<f32> {
    todo!("R3: singular_values")
}

/// R4. Vertex finding (the idea behind VCA) on pixels given as vectors: pick the pixel with
/// the largest length; then repeat: remove from every pixel its part along the directions
/// found so far (project onto their orthogonal complement), and pick the pixel whose
/// remaining part is longest. Returns the indices of the `k` pixels picked.
///
/// Hint: keep a list of orthonormal directions. After picking a pixel, its remaining part
/// divided by its length is the next direction (Gram–Schmidt).
pub fn find_vertices(pixels: &[Vec<f32>], k: usize) -> Vec<usize> {
    todo!("R4: find_vertices")
}

/// R5a. One multiplicative update of the abundances in NMF (slide "The math: the NMF updates"):
/// `H ← H ⊙ (WᵀY) ⊘ (WᵀW H)`, element by element. W is channels × K, H is K × pixels.
///
/// Hint: compute WᵀW (K × K) first. Update the K entries of a pixel (a column of H) from the
/// old values of that column, not from entries you have already changed. Add a tiny number
/// (1e-12) to the denominator to avoid dividing by 0.
pub fn update_h(y: &Matrix, w: &Matrix, h: &mut Matrix) {
    todo!("R5a: update_h")
}

/// R5b. One multiplicative update of the spectra in NMF: `W ← W ⊙ (Y Hᵀ) ⊘ (W H Hᵀ)`.
pub fn update_w(y: &Matrix, w: &mut Matrix, h: &Matrix) {
    todo!("R5b: update_w")
}
