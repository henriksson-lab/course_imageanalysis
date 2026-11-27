// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p shapematch fourier_descriptors::

use imgcourse::{Complex, Rng};
use shapematch::fourier_descriptors::*;

fn z(re: f32, im: f32) -> Complex {
    Complex::new(re, im)
}

fn assert_close(got: Complex, expected: Complex, what: &str) {
    assert!((got - expected).abs() < 1e-4, "{what}: got {got}, expected {expected}");
}

/// A wobbly closed outline (64 points) with no symmetry.
fn wobbly() -> Vec<Complex> {
    let mut rng = Rng::new(3);
    (0..64)
        .map(|i| {
            let t = std::f32::consts::TAU * i as f32 / 64.0;
            let r = 1.0 + 0.3 * (2.0 * t + 0.4).cos() + 0.15 * (3.0 * t).sin() + 0.02 * rng.normal();
            Complex::from_polar(r, t)
        })
        .collect()
}

#[test]
fn c1_square_corners_are_a_circle_b1a() {
    let d = descriptors(&[z(1., 0.), z(0., 1.), z(-1., 0.), z(0., -1.)]);
    assert_close(coef(&d, 1), z(1., 0.), "B1(a) c_1");
    for k in [-1, 0, 2] {
        assert_close(coef(&d, k), z(0., 0.), &format!("B1(a) c_{k}"));
    }
}

#[test]
fn c1_ellipse_b1b_c() {
    let e = [z(2., 0.), z(0., 1.), z(-2., 0.), z(0., -1.)];
    let d = descriptors(&e);
    assert_close(coef(&d, 1), z(1.5, 0.), "B1(b) c_1");
    assert_close(coef(&d, -1), z(0.5, 0.), "B1(b) c_−1");
    assert_close(coef(&d, 0), z(0., 0.), "B1(b) c_0");
    assert_close(coef(&d, 2), z(0., 0.), "B1(b) c_2");
    // (c) turned by 90° (times i) and started one point later
    let rot: Vec<Complex> = e.iter().map(|&p| z(0., 1.) * p).collect();
    let dr = descriptors(&rot);
    assert_close(coef(&dr, 1), z(0., 1.5), "B1(c) turned: c_1");
    assert_close(coef(&dr, -1), z(0., 0.5), "B1(c) turned: c_−1");
    let ds = descriptors(&restart(&e, 1));
    assert_close(coef(&ds, 1), z(0., 1.5), "B1(c) new start: c_1");
    assert_close(coef(&ds, -1), z(0., -0.5), "B1(c) new start: c_−1");
}

#[test]
fn c1_the_rules() {
    let a = wobbly();
    let n = a.len();
    let d = descriptors(&a);
    let alpha = 0.7f32;
    let moved = similarity(&a, 1.0, alpha.to_degrees(), z(3., -2.));
    let dm = descriptors(&moved);
    let dl = descriptors(&restart(&a, 5));
    let dr = descriptors(&mirror_traced(&a));
    for k in -10isize..=10 {
        let ck = coef(&d, k);
        let turned = if k == 0 { Complex::from_angle(alpha) * ck + z(3., -2.) } else { Complex::from_angle(alpha) * ck };
        assert_close(coef(&dm, k), turned, &format!("rotation + translation, k = {k}"));
        let ramp = Complex::from_angle(std::f32::consts::TAU * (k * 5) as f32 / n as f32);
        assert_close(coef(&dl, k), ramp * ck, &format!("start 5 later, k = {k}"));
        assert_close(coef(&dr, k), ck.conj(), &format!("mirror image walked clockwise, k = {k}"));
    }
}

#[test]
fn c2_reconstruct() {
    let a = wobbly();
    let d = descriptors(&a);
    for (p, q) in a.iter().zip(reconstruct(&d, 32)) {
        assert_close(q, *p, "all coefficients give back the samples");
    }
    // only |k| <= 1 of the ellipse (2, i, −2, −i): the ellipse itself
    let e = [z(2., 0.), z(0., 1.), z(-2., 0.), z(0., -1.)];
    for (p, q) in e.iter().zip(reconstruct(&descriptors(&e), 1)) {
        assert_close(q, *p, "K = 1 keeps the ellipse");
    }
    // K = 0: every point at the centroid
    let c0 = coef(&d, 0);
    for q in reconstruct(&d, 0) {
        assert_close(q, c0, "K = 0 is the centroid");
    }
}

#[test]
fn c3_resample_arc_length() {
    // a 4 × 2 rectangle, perimeter 12: 6 points every 2 units
    let r = [z(0., 0.), z(4., 0.), z(4., 2.), z(0., 2.)];
    let s = resample_arc_length(&r, 6);
    let expected = [z(0., 0.), z(2., 0.), z(4., 0.), z(4., 2.), z(2., 2.), z(0., 2.)];
    assert_eq!(s.len(), 6, "6 points");
    for (g, e) in s.iter().zip(expected) {
        assert_close(*g, e, "rectangle resampled every 2 units");
    }
    let s = resample_arc_length(&r, 8);
    assert_close(s[1], z(1.5, 0.), "8 points: every 1.5 units");
    assert_close(s[3], z(4., 0.5), "8 points: sample 3 is around the corner");
}

#[test]
fn c4_magnitudes_are_invariant() {
    let a = wobbly();
    let s = magnitude_signature(&descriptors(&a), 6);
    assert_eq!(s.len(), 12, "k = −6 … −1, 1 … 6: 12 values");
    assert!((s[6] - 1.0).abs() < 1e-5, "|c_1| / |c_1| = 1 (the 7th value)");
    let b = similarity(&restart(&a, 17), 1.8, -50.0, z(4., 4.));
    for (p, q) in s.iter().zip(magnitude_signature(&descriptors(&b), 6)) {
        assert!((p - q).abs() < 1e-4, "turned, scaled, moved and re-started: same magnitudes ({p} vs {q})");
    }
    let e = magnitude_signature(&descriptors(&[z(2., 0.), z(0., 1.), z(-2., 0.), z(0., -1.)]), 1);
    assert!((e[0] - 1.0 / 3.0).abs() < 1e-5, "B1(b): |c_−1| / |c_1| = 1/3, got {}", e[0]);
}

#[test]
fn c5_match_recovers_rotation_scale_and_start() {
    let a = wobbly();
    for (s, deg, m) in [(1.3, 40.0, 20), (0.6, -120.0, 3), (2.0, 170.0, 63)] {
        let b = similarity(&restart(&a, m), s, deg, z(5., 7.));
        let fit = match_outlines(&a, &b, false);
        assert_eq!(fit.m, m, "start m");
        assert!((fit.scale() - s).abs() < 1e-3, "scale {} vs {s}", fit.scale());
        assert!((fit.angle_deg() - deg).abs() < 0.05, "angle {} vs {deg}", fit.angle_deg());
        assert!((fit.score - 1.0).abs() < 1e-4, "score {} should be 1", fit.score);
        assert!(!fit.mirrored);
        for (p, q) in aligned(&a, &fit).iter().zip(&b) {
            assert_close(*p, *q, "A moved onto B");
        }
    }
}

#[test]
fn c5_mirror_image() {
    let a = wobbly();
    let b = similarity(&restart(&mirror_traced(&a), 9), 1.1, 25.0, z(0., 0.));
    let no = match_outlines(&a, &b, false);
    let yes = match_outlines(&a, &b, true);
    assert!(no.misfit() > 0.05, "a mirror image should not fit without the mirror (misfit {})", no.misfit());
    assert!(yes.mirrored && (yes.score - 1.0).abs() < 1e-4, "with the mirror allowed it fits (score {})", yes.score);
    assert!((yes.scale() - 1.1).abs() < 1e-3 && yes.m == 9, "scale {} and start {}", yes.scale(), yes.m);
}

#[test]
fn given_trace_and_fill() {
    // a 3 × 2 block of pixels: 6 boundary pixels, clockwise from the top-left
    let sq = [z(0.5, 0.5), z(3.5, 0.5), z(3.5, 2.5), z(0.5, 2.5)];
    let mask = fill_polygon(&sq, 5, 4);
    let t = trace_boundary(&mask);
    let expected = [z(1., 1.), z(2., 1.), z(3., 1.), z(3., 2.), z(2., 2.), z(1., 2.)];
    assert_eq!(t, expected.to_vec(), "boundary of the block");
}
