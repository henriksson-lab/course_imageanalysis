// Coding exercises for "The structure tensor". Replace each `todo!()` with your
// code, then check it with `cargo test -p edges structure_tensor::` and look at `cargo run -p edges -- structure-tensor`.
#![allow(unused_variables, unused_imports)]

use crate::structure_tensor::{gaussian_blur, gradient};
use imgcourse::Image;

/// C1. The structure tensor of a window, from its gradient vectors: the plain average of
/// the outer products g gᵀ, returned as [Jxx, Jxy, Jyy] = [mean of gx², mean of gx·gy,
/// mean of gy²].
///
/// Hint: three running sums, then divide by `grads.len() as f32`.
pub fn tensor_of_gradients(grads: &[[f32; 2]]) -> [f32; 3] {
    todo!("C1: tensor_of_gradients")
}

/// C2a. The eigenvalues μ1 ≥ μ2 of J = [[a, b], [b, c]] (with j = [a, b, c]), and the angle θ
/// (radians, image coordinates, y down) of the eigenvector of μ1, the dominant gradient
/// direction: θ = ½ atan2(2b, a − c). Returns (μ1, μ2, θ).
///
/// Hint: the closed form of the Hessian unit: (a + c)/2 ± √(((a − c)/2)² + b²); `y.atan2(x)` is
/// atan2(y, x).
pub fn eigen_tensor(j: [f32; 3]) -> (f32, f32, f32) {
    todo!("C2a: eigen_tensor")
}

/// C2b. The coherence ((μ1 − μ2)/(μ1 + μ2))²: 1 = perfectly oriented, 0 = no preferred
/// direction. 0 when both eigenvalues are 0 (a flat window).
pub fn coherence(mu1: f32, mu2: f32) -> f32 {
    todo!("C2b: coherence")
}

/// C3. The Harris corner response det J − k (trace J)²: positive at corners, negative at
/// edges, about 0 in flat regions.
pub fn harris(j: [f32; 3], k: f32) -> f32 {
    todo!("C3: harris")
}

/// C4. The structure tensor at every pixel: the gradient at scale `sigma`, the three products
/// Lx², Lx·Ly and Ly², each averaged with a Gaussian window of width `rho` (no window when
/// `rho` is 0). Returns [Jxx, Jxy, Jyy] per pixel.
///
/// Hint: `gradient(img, sigma)` and `gaussian_blur(img, rho)` are in given.rs; `zip_map`
/// multiplies two images pixel by pixel; `Image::from_fn` puts the three results together.
pub fn structure_tensor(img: &Image<f32>, sigma: f32, rho: f32) -> Image<[f32; 3]> {
    todo!("C4: structure_tensor")
}

/// C5. Corners: the pixels whose response is above `threshold` and larger than all of their
/// 8 neighbours (neighbours outside the image do not count), as (x, y). Ties: where two
/// neighbours have exactly the same response (common in symmetric images), keep the one that
/// comes first row by row; so a pixel must be larger than its neighbours that come before it
/// (the row above, and the one to its left) and at least as large as those after it.
///
/// Hint: as the local maxima of the scale-space unit, in 2D. A neighbour (dx, dy) comes
/// before the pixel if dy < 0, or dy = 0 and dx < 0.
pub fn corners(response: &Image<f32>, threshold: f32) -> Vec<(usize, usize)> {
    todo!("C5: corners")
}
