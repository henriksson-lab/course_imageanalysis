// Coding exercises for "Polynomial wiggles, piecewise cubics and Bézier curves". Replace each
// `todo!()` with your code, then check it with `cargo test -p bezier` and look at `cargo run -p bezier`.
#![allow(unused_variables, unused_imports)]

/// A point of the plane, `[x, y]`.
pub type Point = [f32; 2];

/// C1. The one polynomial through the points `(xs[i], ys[i])`, evaluated at `x`, in the
/// Lagrange form: `sum over i of ys[i] · L_i(x)`, where `L_i` is 1 at `xs[i]` and 0 at the
/// other nodes (slide "The math: the interpolation polynomial").
///
/// Hint: `L_i(x)` is the product over all `j ≠ i` of `(x − xs[j]) / (xs[i] − xs[j])`.
pub fn lagrange(xs: &[f32], ys: &[f32], x: f32) -> f32 {
    todo!("C1: lagrange")
}

/// The point a fraction `t` of the way from `p` to `q`.
pub fn lerp(p: Point, q: Point, t: f32) -> Point {
    [(1.0 - t) * p[0] + t * q[0], (1.0 - t) * p[1] + t * q[1]]
}

/// C2. A point of the Bézier curve with control points `ctrl` (any degree), by de Casteljau:
/// move a fraction `t` along every leg, and repeat on the new, shorter polygon until one
/// point is left (slide "De Casteljau: interpolate, then interpolate again").
///
/// Hint: keep a `Vec<Point>`, start with `ctrl.to_vec()`, and replace it by the `len − 1`
/// points `lerp(pts[i], pts[i + 1], t)` while it has more than one point.
pub fn de_casteljau(ctrl: &[Point], t: f32) -> Point {
    todo!("C2: de_casteljau")
}

/// C3. Split the curve at `t` into two Bézier curves of the same degree: the left one is made
/// of the **first** point of every level of de Casteljau, the right one of the **last** point
/// of every level, from the bottom level back up (slide "What the control polygon tells you").
///
/// Hint: the same loop as C2; collect `next[0]` and `next[next.len() − 1]` at every level.
/// The right half must start at the split point, so reverse it at the end.
pub fn split(ctrl: &[Point], t: f32) -> (Vec<Point>, Vec<Point>) {
    todo!("C3: split")
}

/// C4. The Bernstein weights `b_{i,n}(t) = (n choose i) t^i (1 − t)^(n − i)` for
/// `i = 0, ..., n` (slide "The math: the Bernstein form").
///
/// Hint: `(n choose i+1) = (n choose i) · (n − i) / (i + 1)`, starting from `(n choose 0) = 1`;
/// `t.powi(k)` is `t` to the power `k` (an `i32`).
pub fn bernstein(n: usize, t: f32) -> Vec<f32> {
    todo!("C4: bernstein")
}

/// C5a. The control points of the derivative curve: `n (P_{i+1} − P_i)`, one fewer than
/// `ctrl` (slide "The math: derivatives and the C¹ condition").
pub fn derivative(ctrl: &[Point]) -> Vec<Point> {
    todo!("C5a: derivative")
}

/// C5b. The curvature `κ = (x'y'' − y'x'') / (x'² + y'²)^(3/2)` at `t` (degree 2 or more).
/// Positive: turning left (counter-clockwise, with `y` up).
///
/// Hint: the first derivative is the Bézier curve of `derivative(ctrl)`, the second that of
/// `derivative(&derivative(ctrl))`; evaluate both at `t` with `de_casteljau`.
pub fn curvature(ctrl: &[Point], t: f32) -> f32 {
    todo!("C5b: curvature")
}
