// Coding exercises for "Maximum margin and support vectors". Replace each `todo!()` with your
// code, then check it with `cargo test -p day34 maxmargin::` and look at `cargo run -p day34 -- maxmargin`.
#![allow(unused_variables, unused_imports)]

/// A point in a 2D feature space, `[x1, x2]`. Labels are `+1.0` or `-1.0`.
pub type Point = [f32; 2];

/// C1. The four losses of a point with margin value `m = y f(x)` (slide "Losses: what does
/// the SVM minimise?"): `[0–1, hinge, logistic, squared]`. The logistic loss is in bits,
/// `log2(1 + e^(−m))`.
///
/// Hint: `(1.0 - m).max(0.0)`; `x.log2()` and `x.exp()` for the logistic loss.
pub fn losses(m: f32) -> [f32; 4] {
    todo!("C1: losses")
}

/// C2. The margin values `y_i f(x_i)` of all training points, with `f(x) = w · x + b`.
/// On the edge of the street: 1; inside it: between 0 and 1; on the wrong side: below 0.
pub fn margins(points: &[Point], labels: &[f32], w: Point, b: f32) -> Vec<f32> {
    todo!("C2: margins")
}

/// C3. The soft-margin objective `½‖w‖² + C Σ hinge(y_i f(x_i))` (slide "The math: the
/// soft-margin problem"). Small means a wide street with few violations.
///
/// Hint: use `margins`.
pub fn objective(points: &[Point], labels: &[f32], w: Point, b: f32, c: f32) -> f32 {
    todo!("C3: objective")
}

/// C4. Train a linear soft-margin SVM by subgradient descent on the objective of C3, starting
/// from `w = 0`, `b = 0`. One step: with `V` the points that violate the street
/// (`y f(x) < 1`),
///
/// ```text
/// gradient for w:  w − C Σ_{i in V} y_i x_i
/// gradient for b:     − C Σ_{i in V} y_i
/// ```
///
/// Step `t = 1, 2, ...` moves `(w, b)` a distance `1/√t` against this gradient (the gradient
/// divided by its length, times `1/√t`), so that the steps neither explode for large `C` nor
/// crawl for small `C`. The objective does not go down at every step (the hinge has a kink),
/// so keep the `(w, b)` with the smallest objective seen. Returns `(w, b)`.
///
/// Hint: compute the gradient in one loop over the points; then the length of the
/// whole gradient `(grad_w1, grad_w2, grad_b)`; stop early if it is exactly 0.
pub fn train(points: &[Point], labels: &[f32], c: f32, steps: usize) -> (Point, f32) {
    todo!("C4: train")
}
