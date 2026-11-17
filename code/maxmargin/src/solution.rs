// Model solution for the coding exercises of "Maximum margin and support vectors".

/// A point in a 2D feature space, `[x1, x2]`. Labels are `+1.0` or `-1.0`.
pub type Point = [f32; 2];

/// C1. The four losses of a point with margin value `m = y f(x)` (slide "Losses: what does
/// the SVM minimise?"): `[0–1, hinge, logistic, squared]`. The logistic loss is in bits,
/// `log2(1 + e^(−m))`.
pub fn losses(m: f32) -> [f32; 4] {
    let zero_one = if m <= 0.0 { 1.0 } else { 0.0 };
    let hinge = (1.0 - m).max(0.0);
    let logistic = (1.0 + (-m).exp()).log2();
    let squared = (1.0 - m) * (1.0 - m);
    [zero_one, hinge, logistic, squared]
}

/// C2. The margin values `y_i f(x_i)` of all training points, with `f(x) = w · x + b`.
/// On the edge of the street: 1; inside it: between 0 and 1; on the wrong side: below 0.
pub fn margins(points: &[Point], labels: &[f32], w: Point, b: f32) -> Vec<f32> {
    points.iter().zip(labels).map(|(x, &y)| y * (w[0] * x[0] + w[1] * x[1] + b)).collect()
}

/// C3. The soft-margin objective `½‖w‖² + C Σ hinge(y_i f(x_i))` (slide "The math: the
/// soft-margin problem"). Small means a wide street with few violations.
pub fn objective(points: &[Point], labels: &[f32], w: Point, b: f32, c: f32) -> f32 {
    let mut total = 0.5 * (w[0] * w[0] + w[1] * w[1]);
    for m in margins(points, labels, w, b) {
        total += c * (1.0 - m).max(0.0);
    }
    total
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
pub fn train(points: &[Point], labels: &[f32], c: f32, steps: usize) -> (Point, f32) {
    let (mut w, mut b) = ([0.0f32; 2], 0.0f32);
    let (mut best_w, mut best_b) = (w, b);
    let mut best = objective(points, labels, w, b, c);
    for t in 1..=steps {
        let mut grad_w = w; // from ½‖w‖²
        let mut grad_b = 0.0;
        for (x, &y) in points.iter().zip(labels) {
            if y * (w[0] * x[0] + w[1] * x[1] + b) < 1.0 {
                grad_w[0] -= c * y * x[0];
                grad_w[1] -= c * y * x[1];
                grad_b -= c * y;
            }
        }
        // a step of length 1/√t against the gradient, whatever its size
        let length = (grad_w[0] * grad_w[0] + grad_w[1] * grad_w[1] + grad_b * grad_b).sqrt();
        if length == 0.0 {
            break; // exactly at the optimum
        }
        let rate = 1.0 / ((t as f32).sqrt() * length);
        w = [w[0] - rate * grad_w[0], w[1] - rate * grad_w[1]];
        b -= rate * grad_b;
        let value = objective(points, labels, w, b, c);
        if value < best {
            (best, best_w, best_b) = (value, w, b);
        }
    }
    (best_w, best_b)
}
