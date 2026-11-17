// Coding exercises for "Projections and the sinogram". Replace each `todo!()` with your code,
// then check it with `cargo test -p radon` and look at `cargo run -p radon`.
// Coordinates as in the lecture: origin at the image centre, y up, θ counter-clockwise
// (see `tomo.rs` for the plumbing: `centre`, `sample`, `angles`).
#![allow(unused_variables, unused_imports)]

use crate::tomo::{centre, sample};
use imgcourse::Image;

/// C1. Where the point (x, y) lands on the detector at angle θ (radians):
/// ρ = x cos θ + y sin θ (slide "Lines as (θ, ρ)").
pub fn rho(x: f32, y: f32, theta: f32) -> f32 {
    todo!("C1: rho")
}

/// C2. The projection of an n × n image at angle θ: n detector bins, bin k at ρ = k − c
/// (c = the centre). Each bin is the line integral along its ray, the line of points
/// ρ n + s n⊥ with n = (cos θ, sin θ) and n⊥ = (−sin θ, cos θ): add up the image, read between
/// pixels with `sample`, at the n steps s = j − c, j = 0, 1, ..., n − 1 (step 1, so the sum is
/// the integral).
///
/// Hint: `let c = centre(n);`. The point ρ n + s n⊥ is
/// `(r * cos - s * sin, r * sin + s * cos)`; `sample(img, x, y)` reads the image there.
pub fn projection(img: &Image<f32>, theta: f32) -> Vec<f32> {
    todo!("C2: projection")
}

/// C3. The sinogram: the projections at the given angles side by side. Column a holds the
/// projection at `thetas[a]`, row k the detector bin k: `sino[(a, k)]`.
///
/// Hint: `Image::new(thetas.len(), img.width(), 0.0)`, then one `projection` per column.
pub fn sinogram(img: &Image<f32>, thetas: &[f32]) -> Image<f32> {
    todo!("C3: sinogram")
}
