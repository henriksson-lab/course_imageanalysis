// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day42 colour_vision::

use day42::colour_vision::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// The toy eye of B1 (rows L, M, S over five wavelength bins).
fn toy_eye() -> Vec<Vec<f32>> {
    vec![vec![0., 0., 1., 2., 1.], vec![0., 1., 2., 1., 0.], vec![1., 2., 1., 0., 0.]]
}

#[test]
fn r1_responses_b1() {
    let a = toy_eye();
    for (s, expected) in [([1., 1., 1., 1., 1.], [4., 4., 4.]), ([0., 2., 0., 2., 0.], [4., 4., 4.]), ([2., 0., 1., 0., 2.], [3., 2., 3.])] {
        let c = responses(&a, &s);
        assert!(c.len() == 3 && (0..3).all(|i| close(c[i], expected[i], 1e-5)), "A s for s = {s:?}: got {c:?}, expected {expected:?}");
    }
    let v1 = [3., -2., 1., 0., -1.];
    let c = responses(&a, &v1);
    assert!(c.iter().all(|x| x.abs() < 1e-5), "A v1 (B1b): got {c:?}, expected (0, 0, 0)");
}

#[test]
fn r1_camera_b2() {
    let cam = vec![vec![0., 0., 0., 1., 2.], vec![0., 1., 2., 1., 0.], vec![2., 1., 0., 0., 0.]];
    let c1 = responses(&cam, &[1., 1., 1., 1., 1.]);
    let c2 = responses(&cam, &[0., 2., 0., 2., 0.]);
    assert!(close(c1[0], 3.0, 1e-5) && close(c2[0], 2.0, 1e-5), "camera: got {c1:?} and {c2:?}, expected (3, 4, 3) and (2, 4, 2)");
}

#[test]
fn r2_metamer_b1c() {
    // v1 − v2 = (1, −1, 1, −1, 1) is already invisible; the largest step from s1 is t = 1.
    let s1 = [1., 1., 1., 1., 1.];
    let s = metamer(&toy_eye(), &s1, &[1., -1., 1., -1., 1.], 1.0);
    let expected = [2., 0., 2., 0., 2.];
    assert!((0..5).all(|j| close(s[j], expected[j], 1e-4)), "B1(c) with t = 1: got {s:?}, expected {expected:?}");
    let s = metamer(&toy_eye(), &s1, &[1., -1., 1., -1., 1.], 0.5);
    assert!(close(s[0], 1.5, 1e-4) && close(s[1], 0.5, 1e-4), "B1(c) with t = 0.5: got {s:?}, expected (1.5, 0.5, 1.5, 0.5, 1.5)");
}

#[test]
fn r2_metamer_any_v() {
    // Any v: the result must look the same, be ≥ 0, and (share 1) touch 0 somewhere.
    let a = toy_eye();
    let s1 = [1.0, 0.5, 2.0, 1.5, 0.8];
    let v = [0.3, 2.0, -1.0, 0.7, 1.1];
    let s = metamer(&a, &s1, &v, 1.0);
    let (c, c1) = (responses(&a, &s), responses(&a, &s1));
    for i in 0..3 {
        assert!(close(c[i], c1[i], 1e-3), "response {i}: got {} for the metamer, {} for the original", c[i], c1[i]);
    }
    assert!(s.iter().all(|&x| x > -1e-5), "the metamer has a negative bin: {s:?}");
    assert!(s.iter().any(|&x| x.abs() < 1e-4), "with share 1 some bin should be 0: {s:?}");
    assert!((0..5).any(|j| (s[j] - s1[j]).abs() > 0.1), "the metamer should differ from the original: {s:?}");
}

#[test]
fn r3_chromaticity_b3() {
    let p1 = chromaticity([20., 30., 50.]);
    let p2 = chromaticity([30., 15., 5.]);
    let p = chromaticity([50., 45., 55.]);
    assert!(close(p1[0], 0.2, 1e-5) && close(p1[1], 0.3, 1e-5), "light 1: got {p1:?}, expected (0.2, 0.3)");
    assert!(close(p2[0], 0.6, 1e-5) && close(p2[1], 0.3, 1e-5), "light 2: got {p2:?}, expected (0.6, 0.3)");
    assert!(close(p[0], 1. / 3., 1e-4) && close(p[1], 0.3, 1e-5), "the sum: got {p:?}, expected (0.333, 0.3)");
}

#[test]
fn r4_barycentric_b4() {
    let (r, g, b) = ([0.64, 0.33], [0.30, 0.60], [0.15, 0.06]);
    let c = barycentric([0.30, 0.50], r, g, b);
    let e = [0.067, 0.781, 0.152];
    assert!((0..3).all(|i| close(c[i], e[i], 2e-3)), "B4(a): got {c:?}, expected {e:?} (inside)");
    let c = barycentric([0.20, 0.70], r, g, b);
    let e = [-0.308, 1.339, -0.031];
    assert!((0..3).all(|i| close(c[i], e[i], 2e-3)), "B4(b): got {c:?}, expected {e:?} (outside)");
    let c = barycentric(g, r, g, b);
    assert!(close(c[1], 1.0, 1e-5), "the corner G itself: got {c:?}, expected (0, 1, 0)");
}
