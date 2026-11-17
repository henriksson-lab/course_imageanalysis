// Coding exercises for "Two cell types and an uneven density".
// Replace each `todo!()` with your code, then check it with `cargo test -p day19 marked_patterns::`
// and look at `cargo run -p day19 -- marked-patterns`.
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

/// The translation edge weight `a b / ((a - |dx|)(b - |dy|))` (the ripley-k unit). Given.
pub fn translation_weight(dx: f64, dy: f64, a: f64, b: f64) -> f64 {
    a * b / ((a - dx.abs()) * (b - dy.abs()))
}

/// The standard normal cumulative distribution Φ(z) (a polynomial approximation, accurate
/// to about 1e-7). Given.
pub fn normal_cdf(z: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * z.abs() / std::f64::consts::SQRT_2);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf = 1.0 - poly * (-z * z / 2.0).exp();
    if z >= 0.0 { 0.5 * (1.0 + erf) } else { 0.5 * (1.0 - erf) }
}

/// C1. The cross-K function (slide "The math: cross-K"): type-1 points around type-0 points,
/// `K01 = |W| / (n0 n1) · sum over i in type 0, j in type 1 with d_ij <= r of w_ij`,
/// with the translation weight `w_ij`, or 1 without a correction.
///
/// Hint: one loop over the type-0 points, one over the type-1 points; `translation_weight` is
/// given.
pub fn cross_k(type0: &[Point], type1: &[Point], a: f64, b: f64, r: f64, translation: bool) -> f64 {
    todo!("C1: cross_k")
}

/// C2. Random labelling: the same labels in a random order (Fisher–Yates shuffle: for k from
/// the last index down to 1, swap element k with a random element 0..=k).
///
/// Hint: `rng.below(k + 1)` gives a whole number in 0..=k; `out.swap(k, j)`.
pub fn permute_labels(labels: &[u8], rng: &mut Rng) -> Vec<u8> {
    todo!("C2: permute_labels")
}

/// C3. The mark connection function `p01(r)` (slide "The mark connection function"): of all
/// pairs of points whose distance lies within `h` of `r`, the share whose two labels differ.
/// NaN if there is no such pair.
///
/// Hint: loop over the pairs `i < j` only, so that each pair counts once.
pub fn mark_connection(points: &[Point], labels: &[u8], r: f64, h: f64) -> f64 {
    todo!("C3: mark_connection")
}

/// C4. The inhomogeneous K (slide "The math: the inhomogeneous K") with translation weights:
/// `K = 1/|W| · sum over i != j with d_ij <= r of w_ij / (rho_i rho_j)`, where `rho_i` is the
/// leave-one-out kernel estimate of the intensity at point i: the sum of 2D Gaussians
/// (standard deviation `sigma`, integral 1) of all **other** points, divided by the edge
/// correction `e = [Φ((a - x)/σ) - Φ(-x/σ)] · [Φ((b - y)/σ) - Φ(-y/σ)]`.
///
/// Hint: first the vector of the n intensities `rho[i]` (a loop over the other points, then
/// divide by e), then the double loop over the pairs as in Ripley's K.
pub fn k_inhom(points: &[Point], a: f64, b: f64, r: f64, sigma: f64) -> f64 {
    todo!("C4: k_inhom")
}
