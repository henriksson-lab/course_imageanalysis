// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p two-view

use imgcourse::Rng;
use two_view::*;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// Camera 2 turned by `deg` about the vertical axis; camera coordinates X2 = R X1 + t.
fn rot_y(deg: f64) -> Mat3 {
    let (s, c) = deg.to_radians().sin_cos();
    [[c, 0., s], [0., 1., 0.], [-s, 0., c]]
}

/// Exact matches of `n` points in front of two cameras with focal length `f` and principal
/// point (320, 240); camera 2 has rotation `r` and translation `t`.
fn scene(n: usize, f: f64, r: &Mat3, t: Vec3, seed: u64) -> Vec<Match> {
    let mut rng = Rng::new(seed);
    let proj = |x: Vec3| [f * x[0] / x[2] + 320.0, f * x[1] / x[2] + 240.0];
    (0..n)
        .map(|_| {
            let x = [4.0 * rng.uniform() as f64 - 2.0, 3.0 * rng.uniform() as f64 - 1.5, 4.0 + 4.0 * rng.uniform() as f64];
            let x2 = mat_vec(r, x);
            (proj(x), proj([x2[0] + t[0], x2[1] + t[1], x2[2] + t[2]]))
        })
        .collect()
}

#[test]
fn c1_epipolar_line_b3() {
    let f = [[0., -1., 2.], [1., 0., -3.], [-2., 3., 0.]];
    let l = epipolar_line(&f, [1.0, 1.0]);
    assert!(close(l[0], 1.0, 1e-12) && close(l[1], -2.0, 1e-12) && close(l[2], 1.0, 1e-12), "line of (1, 1): got {l:?}, expected (1, −2, 1)");
    for (x, want) in [([5.0, 3.0], 0.0), ([4.0, 1.0], 3.0 / 5f64.sqrt()), ([2.0, 2.0], 1.0 / 5f64.sqrt()), ([3.0, 2.0], 0.0)] {
        let d = line_distance(l, x);
        assert!(close(d, want, 1e-12), "distance of {x:?} from the line: got {d}, expected {want:.3}");
    }
}

#[test]
fn c2_eight_point_exact() {
    let (r, t) = (rot_y(-10.0), [-1.1, 0.05, 0.4]);
    let matches = scene(30, 700.0, &r, t, 1);
    let f = eight_point(&matches);
    for (k, m) in matches.iter().enumerate() {
        let d = line_distance(epipolar_line(&f, m.0), m.1);
        assert!(d < 1e-6, "match {k}: {d} px from its epipolar line, expected 0 (exact data)");
    }
    let det = f[0][0] * (f[1][1] * f[2][2] - f[1][2] * f[2][1]) - f[0][1] * (f[1][0] * f[2][2] - f[1][2] * f[2][0]) + f[0][2] * (f[1][0] * f[2][1] - f[1][1] * f[2][0]);
    let norm = f.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
    assert!(det.abs() < 1e-9 * norm.powi(3), "F must have rank 2: det F = {det}");
}

#[test]
fn c2_sideways_motion_b2() {
    // B2 (a): R = I and t = (1, 0, 0): the epipolar lines are the rows
    let matches = scene(20, 700.0, &rot_y(0.0), [1.0, 0.0, 0.0], 2);
    let f = eight_point(&matches);
    for m in &matches {
        let l = epipolar_line(&f, m.0);
        let v = -l[2] / l[1];
        assert!(l[0].abs() < 1e-9 * l[1].abs() && close(v, m.0[1], 1e-6), "the epipolar line of {:?} must be the row v = {}: got the line {l:?}", m.0, m.0[1]);
    }
}

#[test]
fn c3_ransac_iterations_b4() {
    let want = [(0.5, 8, 1177), (0.5, 7, 588), (0.5, 5, 146), (0.8, 8, 26), (0.8, 7, 20), (0.8, 5, 12)];
    for (w, s, n) in want {
        let got = ransac_iterations(0.99, w, s);
        assert_eq!(got, n, "w = {w}, s = {s}: got {got}, expected {n}");
    }
}

#[test]
fn c4_ransac_with_outliers() {
    let (r, t) = (rot_y(-10.0), [-1.1, 0.05, 0.4]);
    let mut matches = scene(40, 700.0, &r, t, 3);
    let mut rng = Rng::new(5);
    for m in matches.iter_mut().take(12) {
        m.1 = [640.0 * rng.uniform() as f64, 480.0 * rng.uniform() as f64]; // wrong matches
    }
    let (f, inliers) = ransac_fundamental(&matches, 1.0, 300, &mut rng);
    let good: Vec<usize> = (12..40).collect();
    for k in &good {
        assert!(inliers.contains(k), "the correct match {k} is not among the inliers {inliers:?}");
    }
    for &k in &good {
        let d = line_distance(epipolar_line(&f, matches[k].0), matches[k].1);
        assert!(d < 1e-6, "correct match {k}: {d} px from its epipolar line, expected 0");
    }
    assert!(inliers.len() <= 30, "{} inliers: too many of the 12 wrong matches were accepted", inliers.len());
}

#[test]
fn c5_triangulate_b5() {
    // two parallel cameras 0.1 m apart, f = 800, cx = 320: x = 420 and x' = 400
    let d1 = [(420.0 - 320.0) / 800.0, 0.0, 1.0];
    let d2 = [(400.0 - 320.0) / 800.0, 0.0, 1.0];
    let p = triangulate_midpoint([0.0; 3], d1, [0.1, 0.0, 0.0], d2);
    assert!(close(p[0], 0.5, 1e-9) && close(p[1], 0.0, 1e-9) && close(p[2], 4.0, 1e-9), "got {p:?}, expected (0.5, 0, 4)");
    // two skew rays: the midpoint of the shortest segment
    let p = triangulate_midpoint([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]);
    assert!(close(p[0], 0.0, 1e-9) && close(p[1], 0.5, 1e-9) && close(p[2], 0.0, 1e-9), "rays along x through 0 and along z through (0, 1, 2): got {p:?}, expected (0, 0.5, 0)");
}
