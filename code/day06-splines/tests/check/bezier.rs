// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day06 bezier::

use day06::bezier::*;

/// The worked curve of the lecture and of B2.
const WORKED: [Point; 4] = [[0.0, 0.0], [1.0, 2.0], [3.0, 2.0], [4.0, 0.0]];

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-4
}

fn close_point(p: Point, q: Point) -> bool {
    close(p[0], q[0]) && close(p[1], q[1])
}

#[test]
fn c1_runge_five_points_b1() {
    let xs = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let ys = [0.0, 0.0, 1.0, 0.0, 0.0];
    for i in 0..5 {
        let p = lagrange(&xs, &ys, xs[i]);
        assert!(close(p, ys[i]), "at the node x = {}: got {p}, expected {} (the curve must pass through the samples)", xs[i], ys[i]);
    }
    let p05 = lagrange(&xs, &ys, 0.5);
    let p15 = lagrange(&xs, &ys, 1.5);
    assert!(close(p05, 0.703125), "p(0.5): got {p05}, expected 0.703 (B1b)");
    assert!(close(p15, -0.546875), "p(1.5): got {p15}, expected -0.547 (B1b)");
}

#[test]
fn c2_de_casteljau_b2() {
    let half = de_casteljau(&WORKED, 0.5);
    assert!(close_point(half, [2.0, 1.5]), "B(1/2): got {half:?}, expected (2, 1.5) (B2a)");
    let quarter = de_casteljau(&WORKED, 0.25);
    assert!(close_point(quarter, [0.90625, 1.125]), "B(1/4): got {quarter:?}, expected (0.90625, 1.125) (B2b)");
    assert!(close_point(de_casteljau(&WORKED, 0.0), WORKED[0]), "B(0) must be P0");
    assert!(close_point(de_casteljau(&WORKED, 1.0), WORKED[3]), "B(1) must be P3");
    // Degree 1 is the straight leg itself.
    let line = de_casteljau(&[[0.0, 0.0], [4.0, 2.0]], 0.25);
    assert!(close_point(line, [1.0, 0.5]), "a degree-1 curve at 1/4: got {line:?}, expected (1, 0.5)");
}

#[test]
fn c3_split_b2c() {
    let (left, right) = split(&WORKED, 0.5);
    let left_expected = [[0.0, 0.0], [0.5, 1.0], [1.25, 1.5], [2.0, 1.5]];
    let right_expected = [[2.0, 1.5], [2.75, 1.5], [3.5, 1.0], [4.0, 0.0]];
    assert_eq!(left.len(), 4, "the left half needs 4 control points, got {}", left.len());
    assert_eq!(right.len(), 4, "the right half needs 4 control points, got {}", right.len());
    for i in 0..4 {
        assert!(close_point(left[i], left_expected[i]), "left half: got {left:?}, expected {left_expected:?} (B2c)");
        assert!(close_point(right[i], right_expected[i]), "right half: got {right:?}, expected {right_expected:?} (B2c)");
    }
    // The halves are the same curve: the left half at s is the whole curve at s/2.
    for s in [0.1, 0.3, 0.8] {
        let a = de_casteljau(&left, s);
        let b = de_casteljau(&WORKED, s / 2.0);
        assert!(close_point(a, b), "left half at {s}: got {a:?}, but the whole curve at {} is {b:?}", s / 2.0);
    }
}

#[test]
fn c4_bernstein_b2d() {
    let w = bernstein(3, 0.25);
    let expected = [27.0 / 64.0, 27.0 / 64.0, 9.0 / 64.0, 1.0 / 64.0];
    assert_eq!(w.len(), 4, "a cubic has 4 weights, got {}", w.len());
    for i in 0..4 {
        assert!(close(w[i], expected[i]), "weights at t = 1/4: got {w:?}, expected 27/64, 27/64, 9/64, 1/64 (B2d)");
    }
    let half = bernstein(3, 0.5);
    assert!(close(half[1], 0.375), "weight of P1 at t = 1/2: got {}, expected 3/8", half[1]);
    // Partition of unity, and the Bernstein form gives the same point as de Casteljau.
    for t in [0.0, 0.2, 0.7, 1.0] {
        let sum: f32 = bernstein(6, t).iter().sum();
        assert!(close(sum, 1.0), "degree 6, t = {t}: the weights sum to {sum}, expected 1");
        let b = bernstein(3, t);
        let mut p = [0.0, 0.0];
        for i in 0..4 {
            p[0] += b[i] * WORKED[i][0];
            p[1] += b[i] * WORKED[i][1];
        }
        let q = de_casteljau(&WORKED, t);
        assert!(close_point(p, q), "t = {t}: Bernstein sum {p:?}, de Casteljau {q:?}");
    }
}

#[test]
fn c5_derivative_and_curvature_b10() {
    let d = derivative(&WORKED);
    let expected = [[3.0, 6.0], [6.0, 0.0], [3.0, -6.0]];
    assert_eq!(d.len(), 3, "the derivative of a cubic has 3 control points, got {}", d.len());
    for i in 0..3 {
        assert!(close_point(d[i], expected[i]), "derivative control points: got {d:?}, expected {expected:?}");
    }
    let k0 = curvature(&WORKED, 0.0);
    let k_half = curvature(&WORKED, 0.5);
    assert!((k0 + 0.2385).abs() < 1e-3, "curvature at t = 0: got {k0}, expected -0.239 (B10)");
    assert!((k_half + 0.5926).abs() < 1e-3, "curvature at t = 1/2: got {k_half}, expected -0.593 (B10)");
    // A quarter circle of radius 2 (approximately, with the handles at 0.552): curvature ≈ 1/2.
    let c = 0.5523 * 2.0;
    let quarter = [[2.0, 0.0], [2.0, c], [c, 2.0], [0.0, 2.0]];
    let k = curvature(&quarter, 0.5);
    assert!((k - 0.5).abs() < 0.01, "near-circle of radius 2: got curvature {k}, expected about 0.5 (turning left)");
}
