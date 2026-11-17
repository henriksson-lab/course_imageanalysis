// Model solution for the coding exercises of "Point patterns and complete spatial randomness".

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
pub fn quadrat_counts(points: &[Point], a: f64, b: f64, k: usize) -> Vec<Vec<usize>> {
    let mut counts = vec![vec![0; k]; k];
    for &(x, y) in points {
        // which quadrat; a point on the far edge goes into the last one
        let col = ((x / a * k as f64) as usize).min(k - 1);
        let row = ((y / b * k as f64) as usize).min(k - 1);
        counts[row][col] += 1;
    }
    counts
}

/// C2. The index of dispersion of quadrat counts (slide "Quadrat counts"): returns
/// `(mean, variance, vmr, x2)` with the variance divided by `m - 1`, `vmr = variance / mean`,
/// and the χ² statistic `x2 = sum of (n_k - mean)² / mean = (m - 1) vmr`.
pub fn dispersion(counts: &[Vec<usize>]) -> (f64, f64, f64, f64) {
    let all: Vec<f64> = counts.iter().flatten().map(|&c| c as f64).collect();
    let m = all.len() as f64;
    let mean = all.iter().sum::<f64>() / m;
    let squares: f64 = all.iter().map(|c| (c - mean).powi(2)).sum();
    let variance = squares / (m - 1.0);
    (mean, variance, variance / mean, squares / mean)
}

/// C3. A Thomas cluster pattern (slide "The math: the cluster process"): parents uniform with
/// intensity `kappa` in the window enlarged by `4 sigma` on every side; each parent gets a
/// Poisson(`mu`) number of offspring, displaced by independent normal steps of standard
/// deviation `sigma` in x and y; keep the offspring inside the window.
pub fn thomas(kappa: f64, mu: f64, sigma: f64, a: f64, b: f64, rng: &mut Rng) -> Vec<Point> {
    let margin = 4.0 * sigma;
    let (wa, wb) = (a + 2.0 * margin, b + 2.0 * margin);
    let parents = poisson(kappa * wa * wb, rng);
    let mut pts = Vec::new();
    for _ in 0..parents {
        let (px, py) = (wa * rng.uniform() as f64 - margin, wb * rng.uniform() as f64 - margin);
        for _ in 0..poisson(mu, rng) {
            let (x, y) = (px + sigma * rng.normal() as f64, py + sigma * rng.normal() as f64);
            if (0.0..a).contains(&x) && (0.0..b).contains(&y) {
                pts.push((x, y));
            }
        }
    }
    pts
}

/// C4. The kernel estimate of the intensity, with the edge correction of the slide "The math:
/// kernel intensity", on a grid of 1 µm pixels (pixel `(x, y)` has its centre at
/// `(x + 0.5, y + 0.5)` µm): the sum of 2D Gaussians of standard deviation `sigma` (integral 1)
/// centred on the points, divided by
/// `e = [Φ((a - x)/σ) - Φ(-x/σ)] · [Φ((b - y)/σ) - Φ(-y/σ)]`, the share of a kernel inside W.
pub fn intensity_map(points: &[Point], a: f64, b: f64, sigma: f64) -> Image<f32> {
    let (w, h) = (a.ceil() as usize, b.ceil() as usize);
    let norm = 1.0 / (2.0 * std::f64::consts::PI * sigma * sigma);
    Image::from_fn(w, h, |px, py| {
        let (x, y) = (px as f64 + 0.5, py as f64 + 0.5);
        let mut sum = 0.0;
        for &(xi, yi) in points {
            let d2 = (x - xi).powi(2) + (y - yi).powi(2);
            sum += norm * (-d2 / (2.0 * sigma * sigma)).exp();
        }
        let ex = normal_cdf((a - x) / sigma) - normal_cdf(-x / sigma);
        let ey = normal_cdf((b - y) / sigma) - normal_cdf(-y / sigma);
        (sum / (ex * ey)) as f32
    })
}
