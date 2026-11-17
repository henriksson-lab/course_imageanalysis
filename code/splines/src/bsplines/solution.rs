// Model solution for the coding exercises of "B-splines and NURBS".

/// A point of the plane, `[x, y]`.
pub type Point = [f32; 2];

/// C1. The centred cubic B-spline β³ (four boxes convolved), in closed form
/// (slide "The math: uniform B-splines"): `2/3 − x² + |x|³/2` for `|x| ≤ 1`,
/// `(2 − |x|)³/6` for `1 ≤ |x| ≤ 2`, and 0 further out.
pub fn cubic_bspline(x: f32) -> f32 {
    let a = x.abs();
    if a <= 1.0 {
        2.0 / 3.0 - a * a + 0.5 * a * a * a
    } else if a < 2.0 {
        (2.0 - a).powi(3) / 6.0
    } else {
        0.0
    }
}

/// C2. A point of the **closed** uniform cubic B-spline curve: `C(t) = sum over k of
/// P_(k mod n) β³(t − k)`, so that the bump of control point `k` is centred at `t = k`
/// (slide "Uniform cubic B-spline curves"). Defined for every `t`; it repeats with period `n`.
pub fn closed_cubic(ctrl: &[Point], t: f32) -> Point {
    let n = ctrl.len() as isize;
    let base = t.floor() as isize;
    let mut c = [0.0, 0.0];
    // Only the 4 bumps centred at base − 1, ..., base + 2 are non-zero at t.
    for k in base - 1..=base + 2 {
        let w = cubic_bspline(t - k as f32);
        let p = ctrl[k.rem_euclid(n) as usize];
        c[0] += w * p[0];
        c[1] += w * p[1];
    }
    c
}

/// C3a. The basis function `N_{i,p}(t)` on any knot vector, by Cox–de Boor (slide "The math:
/// Cox–de Boor and continuity"): degree 0 is 1 on `[t_i, t_{i+1})`; degree `p` blends
/// `N_{i,p−1}` (rising ramp) and `N_{i+1,p−1}` (falling ramp), with 0/0 taken as 0.
pub fn basis(i: usize, p: usize, knots: &[f32], t: f32) -> f32 {
    if p == 0 {
        return if knots[i] <= t && t < knots[i + 1] { 1.0 } else { 0.0 };
    }
    let mut value = 0.0;
    let left = knots[i + p] - knots[i];
    if left > 0.0 {
        value += (t - knots[i]) / left * basis(i, p - 1, knots, t);
    }
    let right = knots[i + p + 1] - knots[i + 1];
    if right > 0.0 {
        value += (knots[i + p + 1] - t) / right * basis(i + 1, p - 1, knots, t);
    }
    value
}

/// C3b. A point of the B-spline curve of degree `p`: `C(t) = sum over i of P_i N_{i,p}(t)`.
/// The knot vector has `ctrl.len() + p + 1` entries; the curve is defined for `t` in
/// `[knots[p], knots[ctrl.len()])` (at the very last knot every `N` is 0: stay just below it).
pub fn bspline_point(ctrl: &[Point], p: usize, knots: &[f32], t: f32) -> Point {
    let mut c = [0.0, 0.0];
    for (i, q) in ctrl.iter().enumerate() {
        let n = basis(i, p, knots, t);
        c[0] += n * q[0];
        c[1] += n * q[1];
    }
    c
}

/// C4. A point of the NURBS curve with control points `ctrl` and `weights`: lift every control
/// point to `(w x, w y, w)`, run the ordinary B-spline on the lifted points, and divide by the
/// last coordinate (slide "NURBS: weights in homogeneous coordinates").
pub fn nurbs_point(ctrl: &[Point], weights: &[f32], p: usize, knots: &[f32], t: f32) -> Point {
    let (mut x, mut y, mut w) = (0.0, 0.0, 0.0);
    for i in 0..ctrl.len() {
        let n = basis(i, p, knots, t);
        x += n * weights[i] * ctrl[i][0];
        y += n * weights[i] * ctrl[i][1];
        w += n * weights[i];
    }
    [x / w, y / w]
}
