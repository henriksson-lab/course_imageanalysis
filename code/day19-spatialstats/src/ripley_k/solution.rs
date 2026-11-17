// Model solution for the coding exercises of "Ripley's K, the L function and the pair correlation".

pub use crate::common::{csr, dist, hard_core, poisson, thomas, Point};

/// C1. The translation edge weight of a pair with difference `(dx, dy)` in the window
/// `[0, a] × [0, b]` (slide "The math: why the weights work"):
/// `|W| / |W ∩ (W + v)| = a b / ((a - |dx|) (b - |dy|))`.
pub fn translation_weight(dx: f64, dy: f64, a: f64, b: f64) -> f64 {
    a * b / ((a - dx.abs()) * (b - dy.abs()))
}

/// C2a. The estimate of Ripley's K at distance `r` (slide "The math: the K function"):
/// `K = |W| / (n (n - 1)) · sum over ordered pairs i != j with d_ij <= r of w_ij`, with `w_ij` the
/// translation weight, or 1 without a correction.
pub fn k_hat(points: &[Point], a: f64, b: f64, r: f64, translation: bool) -> f64 {
    let n = points.len() as f64;
    let mut sum = 0.0;
    for (i, &p) in points.iter().enumerate() {
        for (j, &q) in points.iter().enumerate() {
            if i != j && dist(p, q) <= r {
                sum += if translation { translation_weight(q.0 - p.0, q.1 - p.1, a, b) } else { 1.0 };
            }
        }
    }
    a * b / (n * (n - 1.0)) * sum
}

/// C2b. `L(r) - r` with `L = sqrt(K / pi)`: 0 for CSR, above 0 clustered, below 0 regular.
pub fn l_minus_r(k: f64, r: f64) -> f64 {
    (k / std::f64::consts::PI).sqrt() - r
}

/// C3. The estimate of the pair correlation g at distance `r` (slide "The math: pair
/// correlation"), with the Epanechnikov kernel of half-width `h`,
/// `k_h(t) = 3 / (4h) · (1 - (t/h)²)` for `|t| <= h` (0 outside), and translation weights:
/// `g = |W| / (n (n - 1)) · sum over i != j of w_ij k_h(r - d_ij) / (2 pi d_ij)`.
pub fn pair_correlation(points: &[Point], a: f64, b: f64, r: f64, h: f64) -> f64 {
    let n = points.len() as f64;
    let mut sum = 0.0;
    for (i, &p) in points.iter().enumerate() {
        for (j, &q) in points.iter().enumerate() {
            let d = dist(p, q);
            let t = (r - d) / h;
            if i != j && t.abs() <= 1.0 {
                let kernel = 3.0 / (4.0 * h) * (1.0 - t * t);
                sum += translation_weight(q.0 - p.0, q.1 - p.1, a, b) * kernel / (2.0 * std::f64::consts::PI * d);
            }
        }
    }
    a * b / (n * (n - 1.0)) * sum
}
