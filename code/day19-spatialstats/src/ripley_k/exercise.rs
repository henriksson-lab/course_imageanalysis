// Coding exercises for "Ripley's K, the L function and the pair correlation".
// Replace each `todo!()` with your code, then check it with `cargo test -p day19 ripley_k::`
// and look at `cargo run -p day19 -- ripley-k`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Rng;

/// A point `(x, y)` in µm. The window is the rectangle `[0, a] × [0, b]`.
pub type Point = (f64, f64);

/// `n` independent uniform points in the window (CSR given n). Given.
pub fn csr(n: usize, a: f64, b: f64, rng: &mut Rng) -> Vec<Point> {
    (0..n).map(|_| (a * rng.uniform() as f64, b * rng.uniform() as f64)).collect()
}

/// A random Poisson number of the given mean. Given.
pub fn poisson(mean: f64, rng: &mut Rng) -> usize {
    let limit = (-mean).exp();
    let (mut k, mut prod) = (0, rng.uniform() as f64);
    while prod > limit {
        k += 1;
        prod *= rng.uniform() as f64;
    }
    k
}

/// A Thomas cluster pattern (C3 of the point-patterns unit). Given.
pub fn thomas(kappa: f64, mu: f64, sigma: f64, a: f64, b: f64, rng: &mut Rng) -> Vec<Point> {
    let m = 4.0 * sigma;
    let mut pts = Vec::new();
    for _ in 0..poisson(kappa * (a + 2.0 * m) * (b + 2.0 * m), rng) {
        let (px, py) = ((a + 2.0 * m) * rng.uniform() as f64 - m, (b + 2.0 * m) * rng.uniform() as f64 - m);
        for _ in 0..poisson(mu, rng) {
            let (x, y) = (px + sigma * rng.normal() as f64, py + sigma * rng.normal() as f64);
            if (0.0..a).contains(&x) && (0.0..b).contains(&y) {
                pts.push((x, y));
            }
        }
    }
    pts
}

/// A hard-core pattern: up to `n` uniform points, none closer than `h`. Given.
pub fn hard_core(n: usize, h: f64, a: f64, b: f64, rng: &mut Rng) -> Vec<Point> {
    let mut pts: Vec<Point> = Vec::new();
    let mut tries = 0;
    while pts.len() < n && tries < 100 * n {
        tries += 1;
        let p = (a * rng.uniform() as f64, b * rng.uniform() as f64);
        if pts.iter().all(|q| dist(p, *q) >= h) {
            pts.push(p);
        }
    }
    pts
}

/// The distance between two points. Given.
pub fn dist(p: Point, q: Point) -> f64 {
    (p.0 - q.0).hypot(p.1 - q.1)
}

/// C1. The translation edge weight of a pair with difference `(dx, dy)` in the window
/// `[0, a] × [0, b]` (slide "The math: why the weights work"):
/// `|W| / |W ∩ (W + v)| = a b / ((a - |dx|) (b - |dy|))`.
pub fn translation_weight(dx: f64, dy: f64, a: f64, b: f64) -> f64 {
    todo!("C1: translation_weight")
}

/// C2a. The estimate of Ripley's K at distance `r` (slide "The math: the K function"):
/// `K = |W| / (n (n - 1)) · sum over ordered pairs i != j with d_ij <= r of w_ij`, with `w_ij` the
/// translation weight, or 1 without a correction.
///
/// Hint: two loops over all points (`enumerate` gives the index too), skip `i == j`. The
/// difference vector of a pair is `(q.0 - p.0, q.1 - p.1)`.
pub fn k_hat(points: &[Point], a: f64, b: f64, r: f64, translation: bool) -> f64 {
    todo!("C2a: k_hat")
}

/// C2b. `L(r) - r` with `L = sqrt(K / pi)`: 0 for CSR, above 0 clustered, below 0 regular.
pub fn l_minus_r(k: f64, r: f64) -> f64 {
    todo!("C2b: l_minus_r")
}

/// C3. The estimate of the pair correlation g at distance `r` (slide "The math: pair
/// correlation"), with the Epanechnikov kernel of half-width `h`,
/// `k_h(t) = 3 / (4h) · (1 - (t/h)²)` for `|t| <= h` (0 outside), and translation weights:
/// `g = |W| / (n (n - 1)) · sum over i != j of w_ij k_h(r - d_ij) / (2 pi d_ij)`.
///
/// Hint: the same double loop as `k_hat`; a pair contributes only if |r - d| <= h.
pub fn pair_correlation(points: &[Point], a: f64, b: f64, r: f64, h: f64) -> f64 {
    todo!("C3: pair_correlation")
}
