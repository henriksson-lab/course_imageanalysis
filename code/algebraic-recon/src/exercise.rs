// Coding exercises for "Reconstruction as a linear system". Replace each `todo!()` with your
// code, then check it with `cargo test -p algebraic-recon` and look at `cargo run -p algebraic-recon`.
// The matrix A is stored by rows: row i lists the pixels that ray i touches, with their
// weights, as (pixel index, weight). Images are vectors of pixels, row by row.
#![allow(unused_variables, unused_imports)]

use crate::tomo::Row;

/// C1a. The forward projection b = A x: for every ray, the weighted sum of its pixels.
///
/// Hint: `for (i, row) in a.iter().enumerate() { for &(j, w) in row { ... } }`.
pub fn forward(a: &[Row], x: &[f32]) -> Vec<f32> {
    todo!("C1a: forward")
}

/// C1b. The back-projection Aᵀ y: every ray adds its value y_i, with the same weights, to its
/// pixels (slide "Aᵀ is back-projection").
pub fn back(a: &[Row], y: &[f32], n_pixels: usize) -> Vec<f32> {
    todo!("C1b: back")
}

/// C2. One sweep of ART (Kaczmarz) over all rays in order, changing `x` in place: for ray i,
/// error = b_i − a_i·x; add ω · error / ‖a_i‖² · a_i to x (slide "The math: Kaczmarz and
/// SIRT"). With weights 1 this spreads the error evenly over the pixels of the ray.
///
/// Hint: for each row, first compute a_i·x and ‖a_i‖² = Σ w², then update its pixels. Skip a
/// row with ‖a_i‖² = 0.
pub fn art_sweep(a: &[Row], b: &[f32], x: &mut [f32], omega: f32) {
    todo!("C2: art_sweep")
}

/// C3. One SIRT step, changing `x` in place: the error of every ray with the current x, divided
/// by the ray's weight sum, back-projected, and at every pixel divided by the weight sum of the
/// rays through it; times ω.   x ← x + ω W_c Aᵀ W_r (b − A x)
///
/// Hint: compute all errors with the x you start with (`forward`), collect the corrections and
/// the column sums in two vectors, and change x only at the end.
pub fn sirt_step(a: &[Row], b: &[f32], x: &mut [f32], omega: f32) {
    todo!("C3: sirt_step")
}
