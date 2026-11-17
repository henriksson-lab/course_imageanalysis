// Model solution for the coding exercises of "Linear classifiers".

/// A point in a 2D feature space, `[x1, x2]`. Labels are `+1.0` or `-1.0`.
pub type Point = [f32; 2];

/// C1a. The value of the linear classifier, `f(x) = w · x + b` (slide "A threshold is a
/// linear classifier"). The class is the sign of `f`.
pub fn decision(w: Point, b: f32, x: Point) -> f32 {
    w[0] * x[0] + w[1] * x[1] + b
}

/// C1b. The signed distance of `x` to the line `f = 0`: `f(x) / ‖w‖`, positive on the side
/// that `w` points to (slide "The geometry of a line").
pub fn signed_distance(w: Point, b: f32, x: Point) -> f32 {
    let norm = (w[0] * w[0] + w[1] * w[1]).sqrt();
    decision(w, b, x) / norm
}

/// C2. The smallest value of `y · f(x) / ‖w‖` over all training points: the distance of the
/// closest point to the line, negative if some point is on the wrong side.
pub fn smallest_margin(points: &[Point], labels: &[f32], w: Point, b: f32) -> f32 {
    let mut smallest = f32::INFINITY;
    for (x, &y) in points.iter().zip(labels) {
        smallest = smallest.min(y * signed_distance(w, b, *x));
    }
    smallest
}

/// C3. The nearest-class-mean classifier: `w = μ+ − μ−`, and the line through the midpoint
/// of the two means, `b = −w · (μ+ + μ−)/2` (slide "Nearest class mean is a line").
/// Returns `(w, b)`.
pub fn nearest_mean(points: &[Point], labels: &[f32]) -> (Point, f32) {
    let (mut plus, mut minus) = ([0.0; 2], [0.0; 2]);
    let (mut n_plus, mut n_minus) = (0.0, 0.0);
    for (x, &y) in points.iter().zip(labels) {
        if y > 0.0 {
            plus = [plus[0] + x[0], plus[1] + x[1]];
            n_plus += 1.0;
        } else {
            minus = [minus[0] + x[0], minus[1] + x[1]];
            n_minus += 1.0;
        }
    }
    let mu_plus = [plus[0] / n_plus, plus[1] / n_plus];
    let mu_minus = [minus[0] / n_minus, minus[1] / n_minus];
    let w = [mu_plus[0] - mu_minus[0], mu_plus[1] - mu_minus[1]];
    let mid = [(mu_plus[0] + mu_minus[0]) / 2.0, (mu_plus[1] + mu_minus[1]) / 2.0];
    (w, -(w[0] * mid[0] + w[1] * mid[1]))
}

/// C4. The perceptron (slide "One neuron is a linear classifier"): start from `w = 0`,
/// `b = 0`; go through the points in order, again and again; for every point on the wrong
/// side (or on the line), `y f(x) ≤ 0`, update `w ← w + y x`, `b ← b + y`. Stop after a full
/// pass without updates, or after `max_passes` passes. Returns `(w, b, number of updates)`.
pub fn perceptron(points: &[Point], labels: &[f32], max_passes: usize) -> (Point, f32, usize) {
    let (mut w, mut b) = ([0.0f32; 2], 0.0f32);
    let mut updates = 0;
    for _ in 0..max_passes {
        let mut changed = false;
        for (x, &y) in points.iter().zip(labels) {
            if y * decision(w, b, *x) <= 0.0 {
                w = [w[0] + y * x[0], w[1] + y * x[1]];
                b += y;
                updates += 1;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    (w, b, updates)
}

/// The number of training points on the wrong side of the line (or on it).
pub fn count_errors(points: &[Point], labels: &[f32], w: Point, b: f32) -> usize {
    points.iter().zip(labels).filter(|(x, &y)| y * decision(w, b, **x) <= 0.0).count()
}
