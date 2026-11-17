// Coding exercises for "Interpolating, approximating and smoothing splines". Replace each
// `todo!()` with your code, then check it with `cargo test -p day06 spline_fit::` and look at
// `cargo run -p day06 -- spline-fit`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{linalg, Border, Image};

/// A point of the plane, `[x, y]`.
pub type Point = [f32; 2];

/// C1. Solve a tridiagonal system by the Thomas algorithm (Gaussian elimination without
/// pivoting, slide "The math: the natural cubic spline system"). Row `i` reads
/// `lower[i] x[i−1] + diag[i] x[i] + upper[i] x[i+1] = rhs[i]` (`lower[0]` and `upper[n−1]`
/// are not used).
///
/// Hint: work on copies of `diag` and `rhs`. Forward, for `i = 1 .. n−1`: `f = lower[i] / d[i−1]`,
/// `d[i] −= f · upper[i−1]`, `r[i] −= f · r[i−1]`. Then backward from the last row:
/// `x[i] = (r[i] − upper[i] · x[i+1]) / d[i]`.
pub fn solve_tridiagonal(lower: &[f32], diag: &[f32], upper: &[f32], rhs: &[f32]) -> Vec<f32> {
    todo!("C1: solve_tridiagonal")
}

/// C2a. The second derivatives `M_i = s''(x_i)` of the natural cubic spline through `ys`,
/// sampled at `x = 0, 1, 2, ...`: `M_{i−1} + 4 M_i + M_{i+1} = 6 (y_{i−1} − 2 y_i + y_{i+1})`
/// at the interior points, and `M = 0` at both ends.
///
/// Hint: the unknowns are `M_1 ... M_{n−2}`; with `M_0 = M_{n−1} = 0` the system is
/// tridiagonal with 1, 4, 1. Solve it with C1 and put 0 at both ends.
pub fn natural_spline(ys: &[f32]) -> Vec<f32> {
    todo!("C2a: natural_spline")
}

/// C2b. The natural spline at `x` (between 0 and `ys.len() − 1`), from the values and the
/// second derivatives: on `[i, i+1]` with `a = i + 1 − x`, `b = 1 − a`,
/// `s = a y_i + b y_{i+1} + ((a³ − a) M_i + (b³ − b) M_{i+1}) / 6`.
///
/// Hint: `i = floor(x)`, but at most `ys.len() − 2`, so that `x` at the last sample works.
pub fn natural_eval(ys: &[f32], ms: &[f32], x: f32) -> f32 {
    todo!("C2b: natural_eval")
}

/// C3a. The penalty matrix `DᵀD`, where every row of `D` is a second difference
/// `f_{i−1} − 2 f_i + f_{i+1}` (rows `i = 1 ... n−2`). With `closed`, the values lie on a closed
/// curve: `D` has a row for every `i` and the indices wrap around.
///
/// Hint: start from an `n × n` matrix of zeros (`vec![vec![0.0; n]; n]`). Each row of `D` has
/// the weights `w = (1, −2, 1)` at the columns `idx = (i−1, i, i+1)`; it adds `w[a] · w[b]` to
/// `P[idx[a]][idx[b]]` for every `a, b`. Wrap with `(i + n − 1) % n` and `(i + 1) % n`.
pub fn penalty_matrix(n: usize, closed: bool) -> Vec<Vec<f64>> {
    todo!("C3a: penalty_matrix")
}

/// C3b. The discrete smoothing spline: minimise `sum (y_i − f_i)² + λ sum (second difference)²`
/// by solving `(I + λ DᵀD) f = y` (slide "The math: penalised least squares").
///
/// Hint: `linalg::solve(&a, &y)` solves `a x = y` (in `f64`; it returns an `Option`).
pub fn smooth(ys: &[f32], lambda: f32, closed: bool) -> Vec<f32> {
    todo!("C3b: smooth")
}

/// C4. Leave-one-out cross-validation of the (open) discrete smoother without refitting:
/// `LOO = mean of ((y_i − f_i) / (1 − H_ii))²` with the hat matrix `H = (I + λ DᵀD)⁻¹`.
/// Returns `(LOO, tr H)`; `tr H` is the effective number of parameters.
///
/// Hint: the smoother is linear, `f = H y`. So column `i` of `H` is `smooth(e_i)`, the smoother
/// applied to the unit vector `e_i`, and `H_ii` is entry `i` of it.
pub fn loo_cv(ys: &[f32], lambda: f32) -> (f32, f32) {
    todo!("C4: loo_cv")
}

/// The cubic B-spline β³ (from the B-spline unit).
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

/// A point of the closed uniform cubic B-spline with control points `ctrl`; the bump of
/// control point `k` is centred at `t = k` (from the B-spline unit).
pub fn closed_cubic(ctrl: &[Point], t: f32) -> Point {
    let n = ctrl.len() as isize;
    let base = t.floor() as isize;
    let mut c = [0.0, 0.0];
    for k in base - 1..=base + 2 {
        let w = cubic_bspline(t - k as f32);
        let p = ctrl[k.rem_euclid(n) as usize];
        c[0] += w * p[0];
        c[1] += w * p[1];
    }
    c
}

/// C5a. The number of pixel edges between the object (`true`) and the background: the length
/// of the boundary staircase (slide "Smooth outlines from pixels").
///
/// Hint: for every object pixel, count its 4 neighbours that are background; outside the
/// image counts as background: `mask.get_border(x, y, Border::Zero)` gives `false` there.
pub fn edge_count(mask: &Image<bool>) -> usize {
    todo!("C5a: edge_count")
}

/// C5b. Fit a closed cubic B-spline with one control point per data point to the points of an
/// outline (in order around it, data point `i` at `t = i`): minimise
/// `sum |p_i − C(i)|² + λ sum |c_{j−1} − 2 c_j + c_{j+1}|²` by solving
/// `(BᵀB + λ DᵀD) c = Bᵀ p`, once for x and once for y. `B` has the row `1/6, 4/6, 1/6` at
/// `i − 1, i, i + 1` (wrapping around): the curve at the knot `i`.
///
/// Hint: `linalg::transpose`, `linalg::matmul` and `linalg::solve` work on `Vec<Vec<f64>>`;
/// the penalty is `penalty_matrix(n, true)`.
pub fn closed_fit(points: &[Point], lambda: f32) -> Vec<Point> {
    todo!("C5b: closed_fit")
}

/// C5c. The length of the closed curve `closed_cubic(ctrl, t)`, `t` from 0 to `n`, as the length
/// of the polyline through `steps` points per span.
pub fn closed_length(ctrl: &[Point], steps: usize) -> f32 {
    todo!("C5c: closed_length")
}
