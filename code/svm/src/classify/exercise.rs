// Coding exercises for "Linear classifiers". Replace each `todo!()` with your code,
// then check it with `cargo test -p svm classify::` and look at `cargo run -p svm -- classify`.
#![allow(unused_variables, unused_imports)]

/// A point in a 2D feature space, `[x1, x2]`. Labels are `+1.0` or `-1.0`.
pub type Point = [f32; 2];

/// C1a. The value of the linear classifier, `f(x) = w · x + b` (slide "A threshold is a
/// linear classifier"). The class is the sign of `f`.
pub fn decision(w: Point, b: f32, x: Point) -> f32 {
    todo!("C1a: decision")
}

/// C1b. The signed distance of `x` to the line `f = 0`: `f(x) / ‖w‖`, positive on the side
/// that `w` points to (slide "The geometry of a line").
pub fn signed_distance(w: Point, b: f32, x: Point) -> f32 {
    todo!("C1b: signed_distance")
}

/// C2. The smallest value of `y · f(x) / ‖w‖` over all training points: the distance of the
/// closest point to the line, negative if some point is on the wrong side.
///
/// Hint: start with `f32::INFINITY` and keep the minimum; `points.iter().zip(labels)` gives
/// each point together with its label.
pub fn smallest_margin(points: &[Point], labels: &[f32], w: Point, b: f32) -> f32 {
    todo!("C2: smallest_margin")
}

/// C3. The nearest-class-mean classifier: `w = μ+ − μ−`, and the line through the midpoint
/// of the two means, `b = −w · (μ+ + μ−)/2` (slide "Nearest class mean is a line").
/// Returns `(w, b)`.
pub fn nearest_mean(points: &[Point], labels: &[f32]) -> (Point, f32) {
    todo!("C3: nearest_mean")
}

/// C4. The perceptron (slide "One neuron is a linear classifier"): start from `w = 0`,
/// `b = 0`; go through the points in order, again and again; for every point on the wrong
/// side (or on the line), `y f(x) ≤ 0`, update `w ← w + y x`, `b ← b + y`. Stop after a full
/// pass without updates, or after `max_passes` passes. Returns `(w, b, number of updates)`.
///
/// Hint: keep a flag "something changed in this pass".
pub fn perceptron(points: &[Point], labels: &[f32], max_passes: usize) -> (Point, f32, usize) {
    todo!("C4: perceptron")
}

/// The number of training points on the wrong side of the line (or on it). (Given.)
pub fn count_errors(points: &[Point], labels: &[f32], w: Point, b: f32) -> usize {
    points.iter().zip(labels).filter(|(x, &y)| y * decision(w, b, **x) <= 0.0).count()
}
