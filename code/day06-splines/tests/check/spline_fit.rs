// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day06 spline_fit::

use imgcourse::{linalg, Image};
use day06::spline_fit::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_tridiagonal() {
    // 4x0 + x1 = 6, x0 + 4x1 + x2 = 12, x1 + 4x2 + x3 = 18, x2 + 4x3 = 19: solution 1, 2, 3, 4.
    let x = solve_tridiagonal(&[0.0, 1.0, 1.0, 1.0], &[4.0; 4], &[1.0, 1.0, 1.0, 0.0], &[6.0, 12.0, 18.0, 19.0]);
    for (i, &v) in x.iter().enumerate() {
        assert!(close(v, (i + 1) as f32, 1e-4), "unknown {i}: got {v}, expected {} (all: {x:?})", i + 1);
    }
    // Different numbers on every diagonal.
    let x = solve_tridiagonal(&[0.0, 2.0, -1.0], &[3.0, 5.0, 2.0], &[1.0, 1.0, 0.0], &[5.0, 15.0, 4.0]);
    for (i, e) in [1.0, 2.0, 3.0].iter().enumerate() {
        assert!(close(x[i], *e, 1e-4), "got {x:?}, expected [1, 2, 3]");
    }
}

#[test]
fn c2_natural_spline_b2_b7() {
    let ys = [0.0, 1.0, 0.0];
    let m = natural_spline(&ys);
    assert!(close(m[0], 0.0, 1e-5) && close(m[2], 0.0, 1e-5), "natural: M at the ends must be 0, got {m:?}");
    assert!(close(m[1], -3.0, 1e-4), "M_1: got {}, expected -3 (B7)", m[1]);
    for (x, expected) in [(0.0, 0.0), (0.5, 0.6875), (1.0, 1.0), (1.5, 0.6875), (2.0, 0.0)] {
        let s = natural_eval(&ys, &m, x);
        assert!(close(s, expected, 1e-4), "s({x}): got {s}, expected {expected} (B2)");
    }
    // A straight line stays a straight line.
    let line = [1.0, 3.0, 5.0, 7.0, 9.0];
    let m = natural_spline(&line);
    let s = natural_eval(&line, &m, 2.3);
    assert!(close(s, 5.6, 1e-4), "the spline through a line at 2.3: got {s}, expected 5.6");
}

#[test]
fn c3_three_point_smoothing_b1() {
    let ys = [0.0, 3.0, 0.0];
    for (lambda, a, b) in [(1.0, 6.0 / 7.0, 9.0 / 7.0), (0.1, 0.375, 2.25), (10.0, 60.0 / 61.0, 63.0 / 61.0)] {
        let f = smooth(&ys, lambda, false);
        assert!(close(f[0], a, 1e-4) && close(f[1], b, 1e-4) && close(f[2], a, 1e-4), "λ = {lambda}: got {f:?}, expected ({a:.3}, {b:.3}, {a:.3}) (B1)");
        let sum: f32 = f.iter().sum();
        assert!(close(sum, 3.0, 1e-4), "λ = {lambda}: the values sum to {sum}, expected 3 (B1c)");
    }
    let p = penalty_matrix(3, false);
    let expected = [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]];
    for i in 0..3 {
        for j in 0..3 {
            assert!((p[i][j] - expected[i][j]).abs() < 1e-9, "DᵀD for n = 3: got {p:?}, expected {expected:?}");
        }
    }
}

#[test]
fn c3_lines_and_closed_curves() {
    // The open smoother keeps a straight line, whatever λ.
    let line: Vec<f32> = (0..10).map(|i| 2.0 + 0.5 * i as f32).collect();
    let f = smooth(&line, 100.0, false);
    for i in 0..10 {
        assert!(close(f[i], line[i], 1e-3), "a line, λ = 100, sample {i}: got {}, expected {}", f[i], line[i]);
    }
    // The closed penalty wraps around: every row of the closed DᵀD is 1, −4, 6, −4, 1.
    let p = penalty_matrix(8, true);
    let row: Vec<f64> = (0..8).map(|j| p[3][j]).collect();
    assert_eq!(row, vec![0.0, 1.0, -4.0, 6.0, -4.0, 1.0, 0.0, 0.0], "row 3 of the closed DᵀD for n = 8");
    assert_eq!(p[0][7], -4.0, "closed: the first and last values are neighbours, P[0][7] should be -4");
}

/// Leave-one-out by brute force: fit without the data term of point i, predict f_i.
fn loo_brute(ys: &[f32], lambda: f32) -> f32 {
    let n = ys.len();
    let p = penalty_matrix(n, false);
    let mut total = 0.0;
    for i in 0..n {
        let mut a: Vec<Vec<f64>> = p.iter().map(|r| r.iter().map(|v| v * lambda as f64).collect()).collect();
        let mut b = vec![0.0f64; n];
        for k in 0..n {
            if k != i {
                a[k][k] += 1.0;
                b[k] = ys[k] as f64;
            }
        }
        let f = linalg::solve(&a, &b).unwrap();
        total += (ys[i] as f64 - f[i]).powi(2);
    }
    (total / n as f64) as f32
}

#[test]
fn c4_leave_one_out() {
    let ys: Vec<f32> = (0..20).map(|i| (0.4 * i as f32).sin() + if i % 3 == 0 { 0.3 } else { -0.1 }).collect();
    for lambda in [0.1, 1.0, 10.0] {
        let (loo, _) = loo_cv(&ys, lambda);
        let brute = loo_brute(&ys, lambda);
        assert!(close(loo, brute, 1e-3 * brute.max(1e-3)), "λ = {lambda}: LOO by the formula {loo}, by refitting 20 times {brute}");
    }
    let (_, df_small) = loo_cv(&ys, 1e-4);
    let (_, df_large) = loo_cv(&ys, 1e6);
    assert!(close(df_small, 20.0, 0.05), "tr H for a tiny λ: got {df_small}, expected about n = 20");
    assert!(close(df_large, 2.0, 0.05), "tr H for a huge λ: got {df_large}, expected about 2 (a line)");
}

/// A digital disc of radius r on a 64 × 64 grid, centred on the corner (32, 32).
fn disc(r: f32) -> Image<bool> {
    Image::from_fn(64, 64, |x, y| {
        let (dx, dy) = (x as f32 + 0.5 - 32.0, y as f32 + 0.5 - 32.0);
        dx * dx + dy * dy <= r * r
    })
}

#[test]
fn c5_edge_count_b4() {
    assert_eq!(edge_count(&disc(10.0)), 80, "disc of radius 10: 8r = 80 edges (B4a)");
    assert_eq!(edge_count(&disc(3.0)), 24, "disc of radius 3: 8r = 24 edges (B4d)");
    let mut one = Image::new(5, 5, false);
    one[(0, 0)] = true;
    assert_eq!(edge_count(&one), 4, "one pixel in the corner has 4 edges (outside the image counts as background)");
}

#[test]
fn c5_closed_fit_and_length() {
    let n = 64;
    let circle: Vec<Point> = (0..n)
        .map(|i| {
            let a = std::f32::consts::TAU * i as f32 / n as f32;
            [10.0 * a.cos(), 10.0 * a.sin()]
        })
        .collect();
    // λ = 0: the curve passes through the points at the knots.
    let ctrl = closed_fit(&circle, 0.0);
    for i in [0, 10, 33] {
        let c = closed_cubic(&ctrl, i as f32);
        assert!(close(c[0], circle[i][0], 1e-3) && close(c[1], circle[i][1], 1e-3), "λ = 0, point {i}: the curve is at {c:?}, the data at {:?}", circle[i]);
    }
    let length = closed_length(&ctrl, 10);
    assert!(close(length, 62.832, 0.05), "length of the fitted circle of radius 10: got {length}, expected 2π · 10 = 62.83");
    // The closed cubic on the square of B1 stays between 1.833 and 1.886 from the centre (B1c),
    // so its length lies between those of the two circles.
    let square = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
    let l = closed_length(&square, 50);
    assert!(l > 11.5 && l < 11.86, "closed cubic on the square of B1: got length {l}, expected between 2π · 1.833 = 11.52 and 2π · 1.886 = 11.85");
    // More smoothing shrinks the closed curve.
    let stiff = closed_length(&closed_fit(&circle, 1000.0), 10);
    assert!(stiff < length - 0.5, "λ = 1000 should shrink the circle: length {stiff} against {length} for λ = 0");
}
