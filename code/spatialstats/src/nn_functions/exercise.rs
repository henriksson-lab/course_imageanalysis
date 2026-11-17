// Coding exercises for "Nearest neighbours and empty space: G, F and J".
// Replace each `todo!()` with your code, then check it with `cargo test -p spatialstats nn_functions::`
// and look at `cargo run -p spatialstats -- nn-functions`.
#![allow(unused_variables, unused_imports)]

pub use crate::common::{csr, dist, hard_core, poisson, thomas, Point};

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
