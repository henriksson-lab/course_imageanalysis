// Coding exercises for "B-splines and NURBS". Replace each `todo!()` with your code,
// then check it with `cargo test -p bsplines` and look at `cargo run -p bsplines`.
#![allow(unused_variables, unused_imports)]

/// A point of the plane, `[x, y]`.
pub type Point = [f32; 2];

/// C1. The centred cubic B-spline β³ (four boxes convolved), in closed form
/// (slide "The math: uniform B-splines"): `2/3 − x² + |x|³/2` for `|x| ≤ 1`,
/// `(2 − |x|)³/6` for `1 ≤ |x| ≤ 2`, and 0 further out.
pub fn cubic_bspline(x: f32) -> f32 {
    todo!("C1: cubic_bspline")
}

/// C2. A point of the **closed** uniform cubic B-spline curve: `C(t) = sum over k of
/// P_(k mod n) β³(t − k)`, so that the bump of control point `k` is centred at `t = k`
/// (slide "Uniform cubic B-spline curves"). Defined for every `t`; it repeats with period `n`.
///
/// Hint: only the 4 bumps centred at `k = floor(t) − 1, ..., floor(t) + 2` are non-zero.
/// Use `isize` for `k`; `k.rem_euclid(n)` wraps it into `0..n`, also for negative `k`.
pub fn closed_cubic(ctrl: &[Point], t: f32) -> Point {
    todo!("C2: closed_cubic")
}

/// C3a. The basis function `N_{i,p}(t)` on any knot vector, by Cox–de Boor (slide "The math:
/// Cox–de Boor and continuity"): degree 0 is 1 on `[t_i, t_{i+1})`; degree `p` blends
/// `N_{i,p−1}` (rising ramp) and `N_{i+1,p−1}` (falling ramp), with 0/0 taken as 0.
///
/// Hint: a function may call itself (recursion): `basis(i, p - 1, knots, t)`. Skip a ramp
/// whose denominator `knots[..] − knots[..]` is 0 (a repeated knot).
pub fn basis(i: usize, p: usize, knots: &[f32], t: f32) -> f32 {
    todo!("C3a: basis")
}

/// C3b. A point of the B-spline curve of degree `p`: `C(t) = sum over i of P_i N_{i,p}(t)`.
/// The knot vector has `ctrl.len() + p + 1` entries; the curve is defined for `t` in
/// `[knots[p], knots[ctrl.len()])` (at the very last knot every `N` is 0: stay just below it).
pub fn bspline_point(ctrl: &[Point], p: usize, knots: &[f32], t: f32) -> Point {
    todo!("C3b: bspline_point")
}

/// C4. A point of the NURBS curve with control points `ctrl` and `weights`: lift every control
/// point to `(w x, w y, w)`, run the ordinary B-spline on the lifted points, and divide by the
/// last coordinate (slide "NURBS: weights in homogeneous coordinates").
pub fn nurbs_point(ctrl: &[Point], weights: &[f32], p: usize, knots: &[f32], t: f32) -> Point {
    todo!("C4: nurbs_point")
}
