// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day30 shape_models::

use day30::shape_models::*;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// The triangles of B1.
const P: [Point; 3] = [[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]];
const Q: [Point; 3] = [[-1.0, -1.5], [1.4, 0.3], [-1.9, -0.3]];

#[test]
fn c1_procrustes_b1() {
    let s = procrustes(&P, &Q, true);
    assert!(close(s.a, 1.2, 1e-9) && close(s.b, 0.9, 1e-9), "a, b: got {}, {}, expected 1.2, 0.9", s.a, s.b);
    assert!(close(s.t[0], -1.0, 1e-9) && close(s.t[1], -1.5, 1e-9), "t: got {:?}, expected [-1, -1.5]", s.t);
    let err = sum_sq(&apply(s, &P), &Q);
    assert!(err < 1e-12, "the fit should be exact: remaining sum of squares {err}");
}

#[test]
fn c1_rigid_b2() {
    let s = procrustes(&P, &Q, false);
    assert!(close(s.a, 0.8, 1e-9) && close(s.b, 0.6, 1e-9), "rigid a, b: got {}, {}, expected 0.8, 0.6", s.a, s.b);
    let err = sum_sq(&apply(s, &P), &Q);
    assert!(close(err, 0.8333, 1e-3), "rigid fit: remaining sum of squares {err}, expected 0.833");
}

#[test]
fn c2_gpa_of_copies() {
    // One shape in five different positions, sizes and angles: after GPA all coincide.
    let base: Vec<Point> = vec![[0.0, 0.0], [3.0, 0.0], [3.5, 1.0], [1.0, 2.0], [-0.5, 1.0]];
    let poses = [(1.0, 0.0, [0.0, 0.0]), (0.0, 2.0, [5.0, 1.0]), (-0.7, 0.3, [-2.0, 4.0]), (1.5, -1.5, [0.5, 0.5]), (0.2, 0.1, [9.0, -3.0])];
    let shapes: Vec<Vec<Point>> = poses.iter().map(|&(a, b, t)| apply(Similarity { a, b, t }, &base)).collect();
    let (mean, aligned) = gpa(&shapes, 3);
    let c = centroid(&mean);
    assert!(close(c[0], 0.0, 1e-9) && close(c[1], 0.0, 1e-9), "the mean should be centred: centroid {c:?}");
    let size: f64 = mean.iter().map(|p| p[0] * p[0] + p[1] * p[1]).sum();
    assert!(close(size, 1.0, 1e-9), "the mean should have size 1: sum of squares {size}");
    for (k, s) in aligned.iter().enumerate() {
        let err = sum_sq(s, &mean);
        assert!(err < 1e-12, "shape {k} after alignment is {err} away from the mean, expected 0");
    }
}

#[test]
fn c3_segments_b4() {
    let vectors = vec![vec![1.0, 3.0], vec![0.0, 4.0], vec![2.0, 2.0]];
    let (mean, values, modes) = shape_model(&vectors);
    assert!(close(mean[0], 1.0, 1e-9) && close(mean[1], 3.0, 1e-9), "mean: got {mean:?}, expected [1, 3]");
    assert!(close(values[0], 4.0 / 3.0, 1e-9) && close(values[1], 0.0, 1e-9), "eigenvalues: got {values:?}, expected [1.333, 0]");
    let r = std::f64::consts::FRAC_1_SQRT_2;
    let phi = &modes[0];
    assert!(close(phi[0].abs(), r, 1e-6) && close(phi[0], -phi[1], 1e-6), "mode 1: got {phi:?}, expected ±(0.707, −0.707)");
}

#[test]
fn c4_project_and_clamp_b4() {
    let mean = vec![1.0, 3.0];
    let r = std::f64::consts::FRAC_1_SQRT_2;
    let modes = vec![vec![r, -r]];
    let values = vec![4.0 / 3.0];
    let model_shape = |x: &[f64], clamp: bool| generate(&mean, &modes, &project(x, &mean, &modes, &values, 1, clamp));
    let cases: [(&[f64], bool, [f64; 2], &str); 4] = [
        (&[0.5, 3.5], false, [0.5, 3.5], "B4(c)"),
        (&[-4.0, 8.0], false, [-4.0, 8.0], "B4(d), not clamped"),
        (&[-4.0, 8.0], true, [-1.449, 5.449], "B4(d), clamped"),
        (&[2.0, 5.0], false, [0.5, 3.5], "B4(e)"),
    ];
    for (x, clamp, expected, name) in cases {
        let got = model_shape(x, clamp);
        assert!(close(got[0], expected[0], 1e-3) && close(got[1], expected[1], 1e-3), "{name}: model shape {got:?}, expected {expected:?}");
    }
    let b = project(&[0.5, 3.5], &mean, &modes, &values, 1, false);
    assert!(close(b[0], -r, 1e-4), "B4(c): b = {}, expected −0.707", b[0]);
}
