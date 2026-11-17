// Model solution for the coding exercises of "Polynomial wiggles, piecewise cubics and Bézier curves".

/// A point of the plane, `[x, y]`.
pub type Point = [f32; 2];

/// C1. The one polynomial through the points `(xs[i], ys[i])`, evaluated at `x`, in the
/// Lagrange form: `sum over i of ys[i] · L_i(x)`, where `L_i` is 1 at `xs[i]` and 0 at the
/// other nodes (slide "The math: the interpolation polynomial").
pub fn lagrange(xs: &[f32], ys: &[f32], x: f32) -> f32 {
    let mut sum = 0.0;
    for i in 0..xs.len() {
        let mut l = 1.0;
        for j in 0..xs.len() {
            if j != i {
                l *= (x - xs[j]) / (xs[i] - xs[j]);
            }
        }
        sum += ys[i] * l;
    }
    sum
}

/// The point a fraction `t` of the way from `p` to `q`.
pub fn lerp(p: Point, q: Point, t: f32) -> Point {
    [(1.0 - t) * p[0] + t * q[0], (1.0 - t) * p[1] + t * q[1]]
}

/// C2. A point of the Bézier curve with control points `ctrl` (any degree), by de Casteljau:
/// move a fraction `t` along every leg, and repeat on the new, shorter polygon until one
/// point is left (slide "De Casteljau: interpolate, then interpolate again").
pub fn de_casteljau(ctrl: &[Point], t: f32) -> Point {
    let mut pts = ctrl.to_vec();
    while pts.len() > 1 {
        let mut next = Vec::new();
        for i in 0..pts.len() - 1 {
            next.push(lerp(pts[i], pts[i + 1], t));
        }
        pts = next;
    }
    pts[0]
}

/// C3. Split the curve at `t` into two Bézier curves of the same degree: the left one is made
/// of the **first** point of every level of de Casteljau, the right one of the **last** point
/// of every level, from the bottom level back up (slide "What the control polygon tells you").
pub fn split(ctrl: &[Point], t: f32) -> (Vec<Point>, Vec<Point>) {
    let mut left = vec![ctrl[0]];
    let mut right = vec![ctrl[ctrl.len() - 1]];
    let mut pts = ctrl.to_vec();
    while pts.len() > 1 {
        let mut next = Vec::new();
        for i in 0..pts.len() - 1 {
            next.push(lerp(pts[i], pts[i + 1], t));
        }
        left.push(next[0]);
        right.push(next[next.len() - 1]);
        pts = next;
    }
    right.reverse();
    (left, right)
}

/// C4. The Bernstein weights `b_{i,n}(t) = (n choose i) t^i (1 − t)^(n − i)` for
/// `i = 0, ..., n` (slide "The math: the Bernstein form").
pub fn bernstein(n: usize, t: f32) -> Vec<f32> {
    let mut weights = Vec::new();
    let mut binomial = 1.0; // (n choose 0)
    for i in 0..=n {
        weights.push(binomial * t.powi(i as i32) * (1.0 - t).powi((n - i) as i32));
        binomial = binomial * (n - i) as f32 / (i + 1) as f32; // (n choose i+1)
    }
    weights
}

/// C5a. The control points of the derivative curve: `n (P_{i+1} − P_i)`, one fewer than
/// `ctrl` (slide "The math: derivatives and the C¹ condition").
pub fn derivative(ctrl: &[Point]) -> Vec<Point> {
    let n = (ctrl.len() - 1) as f32;
    let mut d = Vec::new();
    for i in 0..ctrl.len() - 1 {
        d.push([n * (ctrl[i + 1][0] - ctrl[i][0]), n * (ctrl[i + 1][1] - ctrl[i][1])]);
    }
    d
}

/// C5b. The curvature `κ = (x'y'' − y'x'') / (x'² + y'²)^(3/2)` at `t` (degree 2 or more).
/// Positive: turning left (counter-clockwise, with `y` up).
pub fn curvature(ctrl: &[Point], t: f32) -> f32 {
    let d1 = derivative(ctrl);
    let d2 = derivative(&d1);
    let v = de_casteljau(&d1, t);
    let a = de_casteljau(&d2, t);
    let speed2 = v[0] * v[0] + v[1] * v[1];
    (v[0] * a[1] - v[1] * a[0]) / speed2.powf(1.5)
}
