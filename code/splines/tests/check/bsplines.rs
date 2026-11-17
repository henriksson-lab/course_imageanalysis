// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p splines bsplines::

use splines::bsplines::*;

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-4
}

fn close_point(p: Point, q: Point) -> bool {
    close(p[0], q[0]) && close(p[1], q[1])
}

#[test]
fn c1_cubic_bspline_values() {
    for (x, expected, name) in [(0.0, 2.0 / 3.0, "2/3"), (1.0, 1.0 / 6.0, "1/6"), (-1.0, 1.0 / 6.0, "1/6"), (0.5, 23.0 / 48.0, "23/48"), (1.5, 1.0 / 48.0, "1/48"), (2.0, 0.0, "0"), (3.0, 0.0, "0")] {
        let got = cubic_bspline(x);
        assert!(close(got, expected), "β³({x}): got {got}, expected {name}");
    }
    // Shifted copies sum to 1 everywhere.
    for x in [0.0, 0.3, 0.5, 0.9] {
        let sum: f32 = (-3..=3).map(|k| cubic_bspline(x - k as f32)).sum();
        assert!(close(sum, 1.0), "sum of shifted bumps at x = {x}: got {sum}, expected 1");
    }
}

#[test]
fn c1_samples_as_control_values_b3() {
    // f(x) = sum of f_k β³(x − k) with the samples 0, 0, 6, 0, 0 (the 6 at k = 2).
    let samples = [0.0, 0.0, 6.0, 0.0, 0.0];
    let f = |x: f32| -> f32 { (0..5).map(|k| samples[k] * cubic_bspline(x - k as f32)).sum() };
    for (x, expected) in [(1.0, 1.0), (2.0, 4.0), (3.0, 1.0), (2.5, 2.875)] {
        assert!(close(f(x), expected), "f({x}): got {}, expected {expected} (B3b)", f(x));
    }
}

#[test]
fn c2_corner_cutting_b1() {
    let square = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
    let expected = [[2.0 / 3.0, 2.0 / 3.0], [10.0 / 3.0, 2.0 / 3.0], [10.0 / 3.0, 10.0 / 3.0], [2.0 / 3.0, 10.0 / 3.0]];
    for k in 0..4 {
        let c = closed_cubic(&square, k as f32);
        assert!(close_point(c, expected[k]), "at the knot of control point {k}: got {c:?}, expected {:?} (B1a)", expected[k]);
    }
    let mid = closed_cubic(&square, 1.5);
    assert!(close_point(mid, [23.0 / 6.0, 2.0]), "halfway between the knots of (4, 0) and (4, 4): got {mid:?}, expected (3.833, 2) (B1b)");
    // Closed: the curve repeats after n.
    let a = closed_cubic(&square, 0.3);
    let b = closed_cubic(&square, 4.3);
    assert!(close_point(a, b), "C(0.3) = {a:?} but C(4.3) = {b:?}: a closed curve repeats");
}

#[test]
fn c3_cox_de_boor_b6() {
    let knots = [0.0, 1.0, 2.0, 3.0];
    for (i, p, t, expected) in [(0, 1, 0.5, 0.5), (0, 1, 1.0, 1.0), (0, 1, 1.5, 0.5), (0, 2, 1.0, 0.5), (0, 2, 1.5, 0.75), (0, 2, 0.5, 0.125), (0, 2, 3.0, 0.0)] {
        let got = basis(i, p, &knots, t);
        assert!(close(got, expected), "N_{i},{p}({t}) on knots 0, 1, 2, 3: got {got}, expected {expected} (B6)");
    }
}

#[test]
fn c3_uniform_knots_give_the_cubic_bump() {
    let knots: Vec<f32> = (0..=13).map(|k| k as f32).collect();
    // N_{i,3} lives on [i, i+4) and is β³ centred at i + 2; N_{5,3} is zero outside [5, 9) (B2a).
    for t in [3.0, 4.2, 5.0, 6.5, 7.0, 8.9, 9.5] {
        let got = basis(5, 3, &knots, t);
        let expected = cubic_bspline(t - 7.0);
        assert!(close(got, expected), "N_5,3({t}): got {got}, expected β³({}) = {expected}", t - 7.0);
    }
    // Partition of unity on the valid range [3, 10).
    for t in [3.0, 4.5, 6.25, 9.99] {
        let sum: f32 = (0..10).map(|i| basis(i, 3, &knots, t)).sum();
        assert!(close(sum, 1.0), "sum of all N_i,3 at t = {t}: got {sum}, expected 1");
    }
}

#[test]
fn c3_clamped_four_points_is_bezier_b10() {
    let knots = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let t = 0.25;
    let expected = [27.0 / 64.0, 27.0 / 64.0, 9.0 / 64.0, 1.0 / 64.0];
    for i in 0..4 {
        let got = basis(i, 3, &knots, t);
        assert!(close(got, expected[i]), "clamped N_{i},3(1/4): got {got}, expected the Bernstein weight {} (B10)", expected[i]);
    }
    let ctrl = [[0.0, 0.0], [1.0, 2.0], [3.0, 2.0], [4.0, 0.0]];
    let c = bspline_point(&ctrl, 3, &knots, 0.5);
    assert!(close_point(c, [2.0, 1.5]), "clamped cubic at t = 1/2: got {c:?}, expected the Bézier point (2, 1.5)");
    let start = bspline_point(&ctrl, 3, &knots, 0.0);
    assert!(close_point(start, ctrl[0]), "a clamped curve starts at P0: got {start:?}");
}

#[test]
fn c4_quarter_circle_b4() {
    let ctrl = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let ones = nurbs_point(&ctrl, &[1.0, 1.0, 1.0], 2, &knots, 0.5);
    assert!(close_point(ones, [0.75, 0.75]), "all weights 1: got {ones:?}, expected (0.75, 0.75) (B4a)");
    let r = std::f32::consts::FRAC_1_SQRT_2;
    let half = nurbs_point(&ctrl, &[1.0, r, 1.0], 2, &knots, 0.5);
    assert!(close_point(half, [r, r]), "weights 1, 1/√2, 1: got {half:?}, expected (0.7071, 0.7071) (B4b)");
    for t in [0.0, 0.1, 0.25, 0.6, 0.9, 0.999] {
        let c = nurbs_point(&ctrl, &[1.0, r, 1.0], 2, &knots, t);
        let radius = (c[0] * c[0] + c[1] * c[1]).sqrt();
        assert!(close(radius, 1.0), "t = {t}: the point {c:?} is at distance {radius} from the centre, expected 1");
    }
    let w15 = nurbs_point(&ctrl, &[1.0, 1.5, 1.0], 2, &knots, 0.5);
    assert!(close_point(w15, [0.8, 0.8]), "middle weight 1.5: got {w15:?}, expected (0.8, 0.8) (B4c)");
}
