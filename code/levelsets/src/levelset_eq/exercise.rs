// Coding exercises for "Moving a curve: the level set equation".
// Replace each `todo!()` with your code, then check it with `cargo test -p levelsets levelset_eq::` and look at
// `cargo run -p levelsets -- levelset-eq`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};

/// C1. The upwind length of the gradient at pixel `(x, y)` (slide "The math: the upwind
/// (Godunov) scheme"), for a front that moves outward (`outward = true`, `F > 0`) or inward.
/// With `D⁻ = φ(x) − φ(x−1)` and `D⁺ = φ(x+1) − φ(x)` (and the same in y):
/// outward: `max(D⁻, 0)² + min(D⁺, 0)²`, inward: `min(D⁻, 0)² + max(D⁺, 0)²`, summed over x
/// and y, then the square root. Outside the image: copy the edge.
///
/// Hint: read the neighbours with `phi.get_border(x as isize - 1, y as isize, Border::Clamp)`;
/// the loop over the two directions can be a loop over the pairs `(D⁻, D⁺)` for x and for y.
pub fn upwind_gradient(phi: &Image<f32>, x: usize, y: usize, outward: bool) -> f32 {
    todo!("C1: upwind_gradient")
}

/// C2. One forward-Euler step of the level set equation `φ_t + F |∇φ| = 0` (slide "One time
/// step: forward Euler"), with the speed `F` given at every pixel: `φ − Δt F |∇φ|`, where
/// `|∇φ|` is the upwind gradient for the sign of `F` at that pixel. Reads only the old φ.
///
/// Hint: `phi.map_xy(|x, y, v| ...)` builds the new image from the old one.
pub fn level_set_step(phi: &Image<f32>, speed: &Image<f32>, dt: f32) -> Image<f32> {
    todo!("C2: level_set_step")
}

/// C3. One step of curvature flow, `F = −εκ` (slide "Speeds: grass fire and curvature flow"):
/// `φ + Δt ε κ |∇φ|`, with central differences, where
/// `κ |∇φ| = (φ_xx φ_y² − 2 φ_x φ_y φ_xy + φ_yy φ_x²) / (φ_x² + φ_y²)` (0 where the gradient is 0).
///
/// Hint: the same differences as `curvature` of the previous unit; here `κ |∇φ|` has the
/// squared length, not its 3/2 power, in the denominator.
pub fn curvature_step(phi: &Image<f32>, eps: f32, dt: f32) -> Image<f32> {
    todo!("C3: curvature_step")
}

/// C4. Reinitialisation by recomputing distances (slide "Keeping φ healthy", method (a)):
/// find every point where the zero level crosses an edge between two neighbouring pixels
/// (linear interpolation, as in the previous unit), then give every pixel the distance to the
/// nearest such point, with the sign of φ. Slow (every pixel against every point), and a
/// little too large between the crossing points (up to about 0.2 pixels), but the zero level
/// stays where it was.
///
/// Hint: first collect the crossing points in a `Vec<[f32; 2]>`, then one pass over all pixels.
pub fn reinitialise(phi: &Image<f32>) -> Image<f32> {
    todo!("C4: reinitialise")
}
