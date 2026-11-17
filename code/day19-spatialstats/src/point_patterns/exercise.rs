// Coding exercises for "Point patterns and complete spatial randomness".
// Replace each `todo!()` with your code, then check it with `cargo test -p day19 point_patterns::`
// and look at `cargo run -p day19 -- point-patterns`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Image, Rng};

pub use crate::common::{csr, hard_core, normal_cdf, poisson, Point};

/// The pattern as a picture: white dots on black, `scale` pixels per µm. Given.
pub fn draw(points: &[Point], a: f64, b: f64, scale: f64) -> Image<f32> {
    let (w, h) = ((a * scale).ceil() as usize, (b * scale).ceil() as usize);
    let mut img = Image::new(w, h, 0.0f32);
    for &(x, y) in points {
        let (cx, cy) = (x * scale, y * scale);
        for (px, py) in img.coords() {
            if (px as f64 + 0.5 - cx).hypot(py as f64 + 0.5 - cy) <= 1.6 {
                img[(px, py)] = 1.0;
            }
        }
    }
    img
}

/// C1. Quadrat counts: divide the window into `k × k` equal rectangles and count the points in
/// each. `counts[row][column]`, row 0 at the top (small y).
///
/// Hint: the column of a point is `(x / a * k as f64) as usize`; use `.min(k - 1)` so that a
/// point on the far edge does not fall outside.
pub fn quadrat_counts(points: &[Point], a: f64, b: f64, k: usize) -> Vec<Vec<usize>> {
    todo!("C1: quadrat_counts")
}

/// C2. The index of dispersion of quadrat counts (slide "Quadrat counts"): returns
/// `(mean, variance, vmr, x2)` with the variance divided by `m - 1`, `vmr = variance / mean`,
/// and the χ² statistic `x2 = sum of (n_k - mean)² / mean = (m - 1) vmr`.
///
/// Hint: `counts.iter().flatten()` runs over all quadrats.
pub fn dispersion(counts: &[Vec<usize>]) -> (f64, f64, f64, f64) {
    todo!("C2: dispersion")
}

/// C3. A Thomas cluster pattern (slide "The math: the cluster process"): parents uniform with
/// intensity `kappa` in the window enlarged by `4 sigma` on every side; each parent gets a
/// Poisson(`mu`) number of offspring, displaced by independent normal steps of standard
/// deviation `sigma` in x and y; keep the offspring inside the window.
///
/// Hint: `poisson(mean, rng)` and `rng.normal()` (as `f64`) are all you need. The number of
/// parents is Poisson with mean κ times the area of the enlarged window.
pub fn thomas(kappa: f64, mu: f64, sigma: f64, a: f64, b: f64, rng: &mut Rng) -> Vec<Point> {
    todo!("C3: thomas")
}

/// C4. The kernel estimate of the intensity, with the edge correction of the slide "The math:
/// kernel intensity", on a grid of 1 µm pixels (pixel `(x, y)` has its centre at
/// `(x + 0.5, y + 0.5)` µm): the sum of 2D Gaussians of standard deviation `sigma` (integral 1)
/// centred on the points, divided by
/// `e = [Φ((a - x)/σ) - Φ(-x/σ)] · [Φ((b - y)/σ) - Φ(-y/σ)]`, the share of a kernel inside W.
///
/// Hint: `Image::from_fn(w, h, |px, py| ...)`; inside, loop over all points. The 2D Gaussian
/// with integral 1 is exp(-d²/(2σ²)) / (2πσ²); `normal_cdf` is given.
pub fn intensity_map(points: &[Point], a: f64, b: f64, sigma: f64) -> Image<f32> {
    todo!("C4: intensity_map")
}
