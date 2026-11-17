// Model solution for the coding exercises of "Interpolating, approximating and smoothing splines".

use crate::common::cubic_bspline;
use imgcourse::{linalg, Border, Image};

/// A point of the plane, `[x, y]`.
pub type Point = [f32; 2];

/// C1. Solve a tridiagonal system by the Thomas algorithm (Gaussian elimination without
/// pivoting, slide "The math: the natural cubic spline system"). Row `i` reads
/// `lower[i] x[i−1] + diag[i] x[i] + upper[i] x[i+1] = rhs[i]` (`lower[0]` and `upper[n−1]`
/// are not used).
pub fn solve_tridiagonal(lower: &[f32], diag: &[f32], upper: &[f32], rhs: &[f32]) -> Vec<f32> {
    let n = diag.len();
    let mut d = diag.to_vec();
    let mut r = rhs.to_vec();
    // Forward: remove lower[i] with row i − 1.
    for i in 1..n {
        let f = lower[i] / d[i - 1];
        d[i] -= f * upper[i - 1];
        r[i] -= f * r[i - 1];
    }
    // Backward substitution.
    let mut x = vec![0.0; n];
    x[n - 1] = r[n - 1] / d[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = (r[i] - upper[i] * x[i + 1]) / d[i];
    }
    x
}

/// C2a. The second derivatives `M_i = s''(x_i)` of the natural cubic spline through `ys`,
/// sampled at `x = 0, 1, 2, ...`: `M_{i−1} + 4 M_i + M_{i+1} = 6 (y_{i−1} − 2 y_i + y_{i+1})`
/// at the interior points, and `M = 0` at both ends.
pub fn natural_spline(ys: &[f32]) -> Vec<f32> {
    let n = ys.len();
    let mut m = vec![0.0; n];
    if n < 3 {
        return m;
    }
    let k = n - 2; // unknowns M_1 ... M_{n−2}
    let rhs: Vec<f32> = (1..n - 1).map(|i| 6.0 * (ys[i - 1] - 2.0 * ys[i] + ys[i + 1])).collect();
    let inner = solve_tridiagonal(&vec![1.0; k], &vec![4.0; k], &vec![1.0; k], &rhs);
    m[1..n - 1].copy_from_slice(&inner);
    m
}

/// C2b. The natural spline at `x` (between 0 and `ys.len() − 1`), from the values and the
/// second derivatives: on `[i, i+1]` with `a = i + 1 − x`, `b = 1 − a`,
/// `s = a y_i + b y_{i+1} + ((a³ − a) M_i + (b³ − b) M_{i+1}) / 6`.
pub fn natural_eval(ys: &[f32], ms: &[f32], x: f32) -> f32 {
    let i = (x.floor().max(0.0) as usize).min(ys.len() - 2);
    let a = (i + 1) as f32 - x;
    let b = 1.0 - a;
    a * ys[i] + b * ys[i + 1] + ((a * a * a - a) * ms[i] + (b * b * b - b) * ms[i + 1]) / 6.0
}

/// C3a. The penalty matrix `DᵀD`, where every row of `D` is a second difference
/// `f_{i−1} − 2 f_i + f_{i+1}` (rows `i = 1 ... n−2`). With `closed`, the values lie on a closed
/// curve: `D` has a row for every `i` and the indices wrap around.
pub fn penalty_matrix(n: usize, closed: bool) -> Vec<Vec<f64>> {
    let mut p = vec![vec![0.0; n]; n];
    let rows: Vec<usize> = if closed { (0..n).collect() } else { (1..n - 1).collect() };
    for i in rows {
        // The row of D: +1, −2, +1 at i − 1, i, i + 1.
        let idx = [(i + n - 1) % n, i, (i + 1) % n];
        let w = [1.0, -2.0, 1.0];
        for a in 0..3 {
            for b in 0..3 {
                p[idx[a]][idx[b]] += w[a] * w[b];
            }
        }
    }
    p
}

/// C3b. The discrete smoothing spline: minimise `sum (y_i − f_i)² + λ sum (second difference)²`
/// by solving `(I + λ DᵀD) f = y` (slide "The math: penalised least squares").
pub fn smooth(ys: &[f32], lambda: f32, closed: bool) -> Vec<f32> {
    let n = ys.len();
    let mut a = penalty_matrix(n, closed);
    for i in 0..n {
        for j in 0..n {
            a[i][j] *= lambda as f64;
        }
        a[i][i] += 1.0;
    }
    let y: Vec<f64> = ys.iter().map(|&v| v as f64).collect();
    let f = linalg::solve(&a, &y).expect("I + λ DᵀD is invertible");
    f.into_iter().map(|v| v as f32).collect()
}

/// C4. Leave-one-out cross-validation of the (open) discrete smoother without refitting:
/// `LOO = mean of ((y_i − f_i) / (1 − H_ii))²` with the hat matrix `H = (I + λ DᵀD)⁻¹`.
/// Returns `(LOO, tr H)`; `tr H` is the effective number of parameters.
pub fn loo_cv(ys: &[f32], lambda: f32) -> (f32, f32) {
    let n = ys.len();
    let f = smooth(ys, lambda, false);
    let mut loo = 0.0;
    let mut trace = 0.0;
    for i in 0..n {
        // Column i of H is the smoother applied to the unit vector e_i; H_ii is its entry i.
        let mut e = vec![0.0; n];
        e[i] = 1.0;
        let hii = smooth(&e, lambda, false)[i];
        trace += hii;
        let r = (ys[i] - f[i]) / (1.0 - hii);
        loo += r * r;
    }
    (loo / n as f32, trace)
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
pub fn edge_count(mask: &Image<bool>) -> usize {
    let mut count = 0;
    for (x, y) in mask.coords() {
        if mask[(x, y)] {
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                if !mask.get_border(x as isize + dx, y as isize + dy, Border::Zero) {
                    count += 1;
                }
            }
        }
    }
    count
}

/// C5b. Fit a closed cubic B-spline with one control point per data point to the points of an
/// outline (in order around it, data point `i` at `t = i`): minimise
/// `sum |p_i − C(i)|² + λ sum |c_{j−1} − 2 c_j + c_{j+1}|²` by solving
/// `(BᵀB + λ DᵀD) c = Bᵀ p`, once for x and once for y. `B` has the row `1/6, 4/6, 1/6` at
/// `i − 1, i, i + 1` (wrapping around): the curve at the knot `i`.
pub fn closed_fit(points: &[Point], lambda: f32) -> Vec<Point> {
    let n = points.len();
    let mut b = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        b[i][(i + n - 1) % n] += 1.0 / 6.0;
        b[i][i] += 4.0 / 6.0;
        b[i][(i + 1) % n] += 1.0 / 6.0;
    }
    let bt = linalg::transpose(&b);
    let mut a = linalg::matmul(&bt, &b);
    let p = penalty_matrix(n, true);
    for i in 0..n {
        for j in 0..n {
            a[i][j] += lambda as f64 * p[i][j];
        }
    }
    let mut ctrl = vec![[0.0; 2]; n];
    for axis in 0..2 {
        let rhs: Vec<f64> = (0..n).map(|i| (0..n).map(|k| bt[i][k] * points[k][axis] as f64).sum()).collect();
        let c = linalg::solve(&a, &rhs).expect("the fit has a unique solution");
        for i in 0..n {
            ctrl[i][axis] = c[i] as f32;
        }
    }
    ctrl
}

/// C5c. The length of the closed curve `closed_cubic(ctrl, t)`, `t` from 0 to `n`, as the length
/// of the polyline through `steps` points per span.
pub fn closed_length(ctrl: &[Point], steps: usize) -> f32 {
    let total = ctrl.len() * steps;
    let mut length = 0.0;
    let mut prev = closed_cubic(ctrl, 0.0);
    for k in 1..=total {
        let c = closed_cubic(ctrl, k as f32 / steps as f32);
        length += ((c[0] - prev[0]).powi(2) + (c[1] - prev[1]).powi(2)).sqrt();
        prev = c;
    }
    length
}
