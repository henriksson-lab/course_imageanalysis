// Coding exercises for "Tracking: speed, MSD and modes of motion". Replace each `todo!()`
// with your code, then check it with `cargo test -p day26 track_analysis::` and look at `cargo run -p day26 -- track-analysis`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Rng;

/// A position `[x, y]` (a 1D track has y = 0).
pub type Point = [f32; 2];

/// The distance between two points (provided).
pub fn distance(a: Point, b: Point) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// C1. Per-track numbers (slide "Per-track numbers"): the path length (sum of the step
/// lengths), the net displacement (first to last point) and the straightness (net / path).
///
/// Hint: `distance(a, b)` (above) is the length of the step from a to b.
pub fn path_stats(track: &[Point]) -> (f32, f32, f32) {
    todo!("C1: path_stats")
}

/// C2. The mean squared displacement for the lags 0, 1, ..., `max_lag` (frames): for each
/// lag, the mean of |r(t + lag) − r(t)|² over all pairs of points that far apart.
/// `result[lag]`; `result[0]` is 0.
///
/// Hint: for lag k there are `track.len() - k` pairs (t, t + k).
pub fn msd(track: &[Point], max_lag: usize) -> Vec<f32> {
    todo!("C2: msd")
}

/// C3a. The least-squares line y = slope · x + intercept through the points (x[i], y[i]).
///
/// Hint: with the means x̄ and ȳ: slope = Σ (x − x̄)(y − ȳ) / Σ (x − x̄)², and the line goes
/// through (x̄, ȳ).
pub fn fit_line(x: &[f32], y: &[f32]) -> (f32, f32) {
    todo!("C3a: fit_line")
}

/// C3b. The least-squares line through the origin, y = slope · x: slope = Σ x y / Σ x².
pub fn fit_through_origin(x: &[f32], y: &[f32]) -> f32 {
    todo!("C3b: fit_through_origin")
}

/// C4. A 2D random walk of `n` positions, starting at (0, 0): every step has independent
/// Gaussian components of variance 2 D Δt. Every recorded position gets its own localisation
/// error, Gaussian with standard deviation `sigma` per axis (0 for none).
///
/// Hint: keep the true position in a variable; push it plus the error
/// (`sigma * rng.normal()` per axis), then move it by `step * rng.normal()` per axis, with
/// step = √(2 D Δt).
pub fn random_walk(n: usize, d: f32, dt: f32, sigma: f32, rng: &mut Rng) -> Vec<Point> {
    todo!("C4: random_walk")
}

/// C5. The apparent speed: the mean step length divided by the frame interval `dt`.
///
/// Hint: n positions have n − 1 steps.
pub fn apparent_speed(track: &[Point], dt: f32) -> f32 {
    todo!("C5: apparent_speed")
}
