// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day49 sfm::

use imgcourse::Rng;
use day49::sfm::*;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_gauss_newton_b2() {
    let z = gauss_newton_depth(0.25, 1.0, 3.0, 3);
    let want = [3.0, 3.75, 3.984375];
    for k in 0..3 {
        assert!(close(z[k], want[k], 1e-9), "after {k} steps: got Z = {}, expected {}", z[k], want[k]);
    }
    assert!(close(z[3], 4.0, 1e-4), "after 3 steps: got Z = {}, expected about 4", z[3]);
}

#[test]
fn c2_project() {
    // a camera at the origin looking along +x: the point (5, −1) is 1 to its right, 5 ahead
    let u = project(&[0.0, 0.0, 0.0], &[5.0, -1.0]);
    assert!(close(u, 130.0 / 5.0 + 80.0, 1e-9), "got u = {u}, expected 106");
    // the same after moving and turning camera and point together
    let (s, c) = 0.7f64.sin_cos();
    let turn = |p: [f64; 2]| [c * p[0] - s * p[1] + 2.0, s * p[0] + c * p[1] - 1.0];
    let u2 = project(&[2.0, -1.0, 0.7], &turn([5.0, -1.0]));
    assert!(close(u2, u, 1e-9), "moved and turned together: got u = {u2}, expected {u}");
    let cams = [[0.0, 0.0, 0.0], [0.0, -4.0, std::f64::consts::FRAC_PI_2]];
    let points = [[5.0, -1.0], [3.0, 1.0]];
    let obs = [Obs { cam: 0, point: 0, u: 110.0 }, Obs { cam: 1, point: 1, u: project(&cams[1], &points[1]) }];
    let r = residuals(&cams, &points, &obs);
    assert!(close(r[0], 4.0, 1e-9) && close(r[1], 0.0, 1e-9), "residuals: got {r:?}, expected (4, 0)");
}

#[test]
fn c3_huber_b6() {
    for (r, cost, pull) in [(0.3, 0.09, 0.6), (-0.2, 0.04, -0.4), (0.4, 0.16, 0.8), (25.0, 49.0, 2.0)] {
        let (c, w) = huber(r, 1.0);
        assert!(close(c, cost, 1e-12), "Huber cost of {r}: got {c}, expected {cost}");
        assert!(close(2.0 * w * r, pull, 1e-12), "Huber pull of {r}: got {}, expected {pull}", 2.0 * w * r);
    }
}

#[test]
fn c4_damped_step() {
    // linear residuals r(δ) = r + J δ: the Gauss–Newton step solves the least-squares problem
    let j = vec![vec![1.0, 0.0], vec![0.0, 2.0], vec![1.0, 1.0]];
    let r = [1.0, -2.0, 0.5];
    let d = damped_step(&j, &r, &[1.0; 3], 0.0).unwrap();
    // normal equations: [[2, 1], [1, 5]] δ = −(1.5, −3.5)
    let want = [-(1.5 * 5.0 + 3.5) / 9.0, (3.5 * 2.0 + 1.5) / 9.0];
    assert!(close(d[0], want[0], 1e-12) && close(d[1], want[1], 1e-12), "Gauss–Newton step: got {d:?}, expected {want:?}");
    // a large μ gives a short step along the scaled negative gradient
    let d = damped_step(&j, &r, &[1.0; 3], 1e6).unwrap();
    assert!(close(d[0] * 1e6, -1.5 / 2.0, 1e-3) && close(d[1] * 1e6, 3.5 / 5.0, 1e-3), "μ = 10⁶: got {d:?}, expected about (−0.75, 0.7) · 10⁻⁶");
    // weight 0 removes an observation
    let d = damped_step(&j, &r, &[1.0, 1.0, 0.0], 0.0).unwrap();
    assert!(close(d[0], -1.0, 1e-12) && close(d[1], 1.0, 1e-12), "with the third weight 0: got {d:?}, expected (−1, 1)");
}

#[test]
fn c5_bundle_adjust_converges() {
    // 6 cameras on a circle looking at 12 points near the centre; exact observations
    let cams: Vec<Cam> = (0..6).map(|j| {
        let a = j as f64 * 1.0;
        [5.0 * a.cos(), 5.0 * a.sin(), a + std::f64::consts::PI]
    }).collect();
    let points: Vec<Pt> = (0..12).map(|i| {
        let a = i as f64 * 0.52;
        [(1.0 + 0.3 * (3.0 * a).sin()) * a.cos(), (1.0 + 0.3 * (3.0 * a).sin()) * a.sin()]
    }).collect();
    let mut obs = Vec::new();
    for (j, c) in cams.iter().enumerate() {
        for (i, p) in points.iter().enumerate() {
            obs.push(Obs { cam: j, point: i, u: project(c, p) });
        }
    }
    let mut rng = Rng::new(4);
    let mut c = cams.clone();
    let mut p = points.clone();
    for (k, cam) in c.iter_mut().enumerate() {
        for v in cam.iter_mut() {
            if k >= 2 {
                *v += 0.05 * rng.normal() as f64;
            }
        }
    }
    for pt in p.iter_mut() {
        for v in pt.iter_mut() {
            *v += 0.05 * rng.normal() as f64;
        }
    }
    let history = bundle_adjust(&mut c, &mut p, &obs, 10, true, None);
    assert!(history[0] > 1.0, "the start should be off: RMS {}", history[0]);
    assert!(history.last().unwrap() < &1e-6, "LM must fit exact data: RMS history {history:?}");
    for w in history.windows(2) {
        assert!(w[1] <= w[0] + 1e-12, "LM must never increase the error: {history:?}");
    }
    for (i, (a, b)) in p.iter().zip(&points).enumerate() {
        assert!(close(a[0], b[0], 1e-5) && close(a[1], b[1], 1e-5), "point {i}: got {a:?}, expected {b:?}");
    }
}
