// Coding exercises for "Nearest neighbours and empty space: G, F and J".
// Replace each `todo!()` with your code, then check it with `cargo test -p nn-functions`
// and look at `cargo run -p nn-functions`.
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

/// A Thomas cluster pattern (the previous unit's C3). Given.
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

/// The distance from `p` to the border of the window `[0, a] × [0, b]`. Given.
pub fn border_distance(p: Point, a: f64, b: f64) -> f64 {
    p.0.min(a - p.0).min(p.1).min(b - p.1)
}

/// Test locations on a grid with spacing `step`, the first at `(step/2, step/2)`. Given.
pub fn grid(a: f64, b: f64, step: f64) -> Vec<Point> {
    let (nx, ny) = ((a / step) as usize, (b / step) as usize);
    let mut pts = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            pts.push(((i as f64 + 0.5) * step, (j as f64 + 0.5) * step));
        }
    }
    pts
}

/// C1. For every point, the distance to its nearest neighbour (another point of the pattern).
///
/// Hint: start every distance at `f64::INFINITY` and take the minimum over all other points
/// (`dist` is given).
pub fn nn_distances(points: &[Point]) -> Vec<f64> {
    todo!("C1: nn_distances")
}

/// C2. The estimate of G(r): the share of points whose nearest neighbour is within `r`.
/// With `border`, only the points at least `r` from the border of the window count (the
/// reduced-sample estimator of the slide "The math: the reduced-sample estimator").
/// Returns NaN if no point is kept.
///
/// Hint: `border_distance(p, a, b)` is given. Count the kept points and, among them, those with
/// `d <= r`.
pub fn g_hat(points: &[Point], a: f64, b: f64, r: f64, border: bool) -> f64 {
    todo!("C2: g_hat")
}

/// C3. The estimate of F(r): the share of the test locations whose nearest point is within `r`;
/// with `border`, only the test locations at least `r` from the border count. NaN if none is kept.
///
/// Hint: the same as `g_hat`, but the distance is from a test location to the nearest point of
/// the pattern.
pub fn f_hat(points: &[Point], tests: &[Point], a: f64, b: f64, r: f64, border: bool) -> f64 {
    todo!("C3: f_hat")
}

/// C4. The estimate of J(r) = (1 - G(r)) / (1 - F(r)), both with the border rule, F from test
/// locations on a grid with spacing `step`.
///
/// Hint: `grid(a, b, step)` makes the test locations.
pub fn j_hat(points: &[Point], a: f64, b: f64, r: f64, step: f64) -> f64 {
    todo!("C4: j_hat")
}
