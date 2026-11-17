// Coding exercises for "From a spectrum to three numbers". Replace each `todo!()` with your
// code, then check it with `cargo test -p colour-vision` and look at `cargo run -p colour-vision`.
#![allow(unused_variables, unused_imports)]

use imgcourse::linalg;

/// R1. The responses of an observer (eye, camera) to a sampled spectrum: one weighted sum
/// per row of `a`, `c[i] = Σ_j a[i][j] · s[j]` (the matrix–vector product c = A s).
pub fn responses(a: &[Vec<f32>], s: &[f32]) -> Vec<f32> {
    todo!("R1: responses")
}

/// R2. A metamer of the spectrum `s` for the observer `a` (3 rows). First remove from `v`
/// everything the observer can see, keeping its part in the null space of `a`:
/// `p = v − Aᵀ u` with `(A Aᵀ) u = A v` (slide "The math: metamers live in the null space").
/// Then step from `s` along `p` as far as the spectrum stays ≥ 0 everywhere, and take the
/// fraction `share` (between 0 and 1) of that step: `s + share · t_max · p`.
///
/// Hint: `linalg::solve(&aat, &av)` solves the 3 × 3 system (in f64; convert with
/// `as f64` and `as f32`). Only the bins where `p[j] < 0` limit the step: `t ≤ s[j] / −p[j]`.
pub fn metamer(a: &[Vec<f32>], s: &[f32], v: &[f32], share: f32) -> Vec<f32> {
    todo!("R2: metamer")
}

/// R3. The chromaticity (x, y) of a colour (X, Y, Z): divide by X + Y + Z.
pub fn chromaticity(xyz: [f32; 3]) -> [f32; 2] {
    todo!("R3: chromaticity")
}

/// R4. The barycentric coordinates (α, β, γ) of the point `p` in the triangle with corners
/// `r`, `g`, `b`: `p = α r + β g + γ b` with α + β + γ = 1. The point is inside the triangle
/// (in the gamut) when all three are ≥ 0.
///
/// Hint: with γ = 1 − α − β the condition becomes two equations in α and β:
/// `α (r − b) + β (g − b) = p − b`. Solve this 2 × 2 system with the determinant.
pub fn barycentric(p: [f32; 2], r: [f32; 2], g: [f32; 2], b: [f32; 2]) -> [f32; 3] {
    todo!("R4: barycentric")
}
