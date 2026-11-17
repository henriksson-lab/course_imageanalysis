// Model solution for the coding exercises of "Projections and the sinogram".
// Coordinates as in the lecture: origin at the image centre, y up, θ counter-clockwise
// (see `tomo.rs` for the plumbing: `centre`, `sample`, `angles`).

use crate::radon::tomo::{centre, sample};
use imgcourse::Image;

/// C1. Where the point (x, y) lands on the detector at angle θ (radians):
/// ρ = x cos θ + y sin θ (slide "Lines as (θ, ρ)").
pub fn rho(x: f32, y: f32, theta: f32) -> f32 {
    x * theta.cos() + y * theta.sin()
}

/// C2. The projection of an n × n image at angle θ: n detector bins, bin k at ρ = k − c
/// (c = the centre). Each bin is the line integral along its ray, the line of points
/// ρ n + s n⊥ with n = (cos θ, sin θ) and n⊥ = (−sin θ, cos θ): add up the image, read between
/// pixels with `sample`, at the n steps s = j − c, j = 0, 1, ..., n − 1 (step 1, so the sum is
/// the integral).
pub fn projection(img: &Image<f32>, theta: f32) -> Vec<f32> {
    let n = img.width();
    let c = centre(n);
    let (cos, sin) = (theta.cos(), theta.sin());
    let mut p = vec![0.0; n];
    for k in 0..n {
        let r = k as f32 - c;
        let mut sum = 0.0;
        for j in 0..n {
            let s = j as f32 - c;
            // the point at distance r along the normal and s along the ray
            sum += sample(img, r * cos - s * sin, r * sin + s * cos);
        }
        p[k] = sum;
    }
    p
}

/// C3. The sinogram: the projections at the given angles side by side. Column a holds the
/// projection at `thetas[a]`, row k the detector bin k: `sino[(a, k)]`.
pub fn sinogram(img: &Image<f32>, thetas: &[f32]) -> Image<f32> {
    let mut sino = Image::new(thetas.len(), img.width(), 0.0);
    for (a, &theta) in thetas.iter().enumerate() {
        for (k, v) in projection(img, theta).into_iter().enumerate() {
            sino[(a, k)] = v;
        }
    }
    sino
}
