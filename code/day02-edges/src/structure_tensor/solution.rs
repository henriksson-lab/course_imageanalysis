// Model solution for the coding exercises of "The structure tensor".

use crate::structure_tensor::{gaussian_blur, gradient};
use imgcourse::Image;

/// C1. The structure tensor of a window, from its gradient vectors: the plain average of
/// the outer products g gᵀ, returned as [Jxx, Jxy, Jyy] = [mean of gx², mean of gx·gy,
/// mean of gy²].
pub fn tensor_of_gradients(grads: &[[f32; 2]]) -> [f32; 3] {
    let mut j = [0.0f32; 3];
    for g in grads {
        j[0] += g[0] * g[0];
        j[1] += g[0] * g[1];
        j[2] += g[1] * g[1];
    }
    let n = grads.len() as f32;
    j.map(|v| v / n)
}

/// C2a. The eigenvalues μ1 ≥ μ2 of J = [[a, b], [b, c]] (with j = [a, b, c]), and the angle θ
/// (radians, image coordinates, y down) of the eigenvector of μ1, the dominant gradient
/// direction: θ = ½ atan2(2b, a − c). Returns (μ1, μ2, θ).
pub fn eigen_tensor(j: [f32; 3]) -> (f32, f32, f32) {
    let [a, b, c] = j;
    let mean = 0.5 * (a + c);
    let root = (0.25 * (a - c) * (a - c) + b * b).sqrt();
    let theta = 0.5 * (2.0 * b).atan2(a - c);
    (mean + root, mean - root, theta)
}

/// C2b. The coherence ((μ1 − μ2)/(μ1 + μ2))²: 1 = perfectly oriented, 0 = no preferred
/// direction. 0 when both eigenvalues are 0 (a flat window).
pub fn coherence(mu1: f32, mu2: f32) -> f32 {
    if mu1 + mu2 <= 0.0 {
        return 0.0;
    }
    let r = (mu1 - mu2) / (mu1 + mu2);
    r * r
}

/// C3. The Harris corner response det J − k (trace J)²: positive at corners, negative at
/// edges, about 0 in flat regions.
pub fn harris(j: [f32; 3], k: f32) -> f32 {
    let [a, b, c] = j;
    let det = a * c - b * b;
    let trace = a + c;
    det - k * trace * trace
}

/// C4. The structure tensor at every pixel: the gradient at scale `sigma`, the three products
/// Lx², Lx·Ly and Ly², each averaged with a Gaussian window of width `rho` (no window when
/// `rho` is 0). Returns [Jxx, Jxy, Jyy] per pixel.
pub fn structure_tensor(img: &Image<f32>, sigma: f32, rho: f32) -> Image<[f32; 3]> {
    let (lx, ly) = gradient(img, sigma);
    let products = [lx.zip_map(&lx, |a, b| a * b), lx.zip_map(&ly, |a, b| a * b), ly.zip_map(&ly, |a, b| a * b)];
    let [jxx, jxy, jyy] = products.map(|p| if rho > 0.0 { gaussian_blur(&p, rho) } else { p });
    Image::from_fn(img.width(), img.height(), |x, y| [jxx[(x, y)], jxy[(x, y)], jyy[(x, y)]])
}

/// C5. Corners: the pixels whose response is above `threshold` and larger than all of their
/// 8 neighbours (neighbours outside the image do not count), as (x, y). Ties: where two
/// neighbours have exactly the same response (common in symmetric images), keep the one that
/// comes first row by row; so a pixel must be larger than its neighbours that come before it
/// (the row above, and the one to its left) and at least as large as those after it.
pub fn corners(response: &Image<f32>, threshold: f32) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (x, y) in response.coords() {
        let v = response[(x, y)];
        if v <= threshold {
            continue;
        }
        let mut is_max = true;
        for dy in -1..=1isize {
            for dx in -1..=1isize {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if (dx, dy) == (0, 0) || !response.contains(nx, ny) {
                    continue;
                }
                let n = response[(nx as usize, ny as usize)];
                let before = dy < 0 || (dy == 0 && dx < 0);
                if n > v || (before && n == v) {
                    is_max = false;
                }
            }
        }
        if is_max {
            out.push((x, y));
        }
    }
    out
}
