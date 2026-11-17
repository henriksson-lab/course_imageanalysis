// Model solution for the coding exercises of "Tracking: speed, MSD and modes of motion".

use imgcourse::Rng;

/// A position `[x, y]` (a 1D track has y = 0).
pub type Point = [f32; 2];

/// The distance between two points (provided).
pub fn distance(a: Point, b: Point) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// C1. Per-track numbers (slide "Per-track numbers"): the path length (sum of the step
/// lengths), the net displacement (first to last point) and the straightness (net / path).
pub fn path_stats(track: &[Point]) -> (f32, f32, f32) {
    let mut path = 0.0;
    for k in 1..track.len() {
        path += distance(track[k - 1], track[k]);
    }
    let net = distance(track[0], track[track.len() - 1]);
    (path, net, net / path)
}

/// C2. The mean squared displacement for the lags 0, 1, ..., `max_lag` (frames): for each
/// lag, the mean of |r(t + lag) − r(t)|² over all pairs of points that far apart.
/// `result[lag]`; `result[0]` is 0.
pub fn msd(track: &[Point], max_lag: usize) -> Vec<f32> {
    let mut result = vec![0.0; max_lag + 1];
    for lag in 1..=max_lag {
        let mut sum = 0.0;
        let pairs = track.len() - lag;
        for t in 0..pairs {
            sum += distance(track[t], track[t + lag]).powi(2);
        }
        result[lag] = sum / pairs as f32;
    }
    result
}

/// C3a. The least-squares line y = slope · x + intercept through the points (x[i], y[i]).
pub fn fit_line(x: &[f32], y: &[f32]) -> (f32, f32) {
    let n = x.len() as f32;
    let mx = x.iter().sum::<f32>() / n;
    let my = y.iter().sum::<f32>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for i in 0..x.len() {
        sxy += (x[i] - mx) * (y[i] - my);
        sxx += (x[i] - mx) * (x[i] - mx);
    }
    let slope = sxy / sxx;
    (slope, my - slope * mx)
}

/// C3b. The least-squares line through the origin, y = slope · x: slope = Σ x y / Σ x².
pub fn fit_through_origin(x: &[f32], y: &[f32]) -> f32 {
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for i in 0..x.len() {
        sxy += x[i] * y[i];
        sxx += x[i] * x[i];
    }
    sxy / sxx
}

/// C4. A 2D random walk of `n` positions, starting at (0, 0): every step has independent
/// Gaussian components of variance 2 D Δt. Every recorded position gets its own localisation
/// error, Gaussian with standard deviation `sigma` per axis (0 for none).
pub fn random_walk(n: usize, d: f32, dt: f32, sigma: f32, rng: &mut Rng) -> Vec<Point> {
    let step = (2.0 * d * dt).sqrt();
    let mut true_pos = [0.0f32; 2];
    let mut track = Vec::new();
    for _ in 0..n {
        track.push([true_pos[0] + sigma * rng.normal(), true_pos[1] + sigma * rng.normal()]);
        true_pos[0] += step * rng.normal();
        true_pos[1] += step * rng.normal();
    }
    track
}

/// C5. The apparent speed: the mean step length divided by the frame interval `dt`.
pub fn apparent_speed(track: &[Point], dt: f32) -> f32 {
    let mut sum = 0.0;
    for k in 1..track.len() {
        sum += distance(track[k - 1], track[k]);
    }
    sum / (track.len() - 1) as f32 / dt
}
