// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day18 moment_invariants::

use imgcourse::{samples, Image};
use day18::moment_invariants::*;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// An L-shaped mask, 40 × 40.
fn ell() -> Image<f32> {
    Image::from_fn(40, 40, |x, y| if (5..13).contains(&x) && (4..34).contains(&y) || (5..30).contains(&x) && (26..34).contains(&y) { 1.0 } else { 0.0 })
}

/// The three equal point masses of B3, (3, 1), (-1, 2), (-2, -3) with y up, put into a
/// 7 × 7 image with the origin at pixel (3, 3): column = 3 + x, row = 3 - y.
fn b3() -> Image<f32> {
    let mut img = Image::new(7, 7, 0.0);
    for (x, y) in [(3i32, 1i32), (-1, 2), (-2, -3)] {
        img[((3 + x) as usize, (3 - y) as usize)] = 1.0;
    }
    img
}

#[test]
fn c1_central_from_raw_equals_direct() {
    let img = samples::cell().crop(100, 60, 30, 20);
    let mu = central_from_raw(&raw_moments(&img, 4));
    let m = raw_moments(&img, 1);
    let (xbar, ybar) = (m[1][0] / m[0][0], m[0][1] / m[0][0]);
    for p in 0..=4usize {
        for q in 0..=4 - p {
            let direct: f64 = img.coords().map(|(x, y)| (x as f64 - xbar).powi(p as i32) * (y as f64 - ybar).powi(q as i32) * img[(x, y)] as f64).sum();
            assert!(close(mu[p][q], direct, 1e-6 * direct.abs().max(1.0)), "mu{p}{q}: from raw moments {}, directly {}", mu[p][q], direct);
        }
    }
}

#[test]
fn c1_mu30_b8() {
    // mu30 = m30 - 3 xbar m20 + 2 xbar² m10 for a small table of raw moments
    let mut m = vec![vec![0.0; 4]; 4];
    m[0][0] = 4.0;
    m[1][0] = 6.0;
    m[0][1] = 2.0;
    m[2][0] = 14.0;
    m[3][0] = 36.0;
    let mu = central_from_raw(&m);
    let xbar = 1.5;
    let expected = 36.0 - 3.0 * xbar * 14.0 + 2.0 * xbar * xbar * 6.0;
    assert!(close(mu[3][0], expected, 1e-9), "mu30: got {}, expected {expected}", mu[3][0]);
}

#[test]
fn c2_normalised_rectangle_b1() {
    // the 4 × 2 rectangle: mu00 = 8, mu20 = 32/3, mu02 = 8/3; and the 12 × 6 one
    for (a, b) in [(4.0f64, 2.0f64), (12.0, 6.0)] {
        let mut mu = vec![vec![0.0; 3]; 3];
        mu[0][0] = a * b;
        mu[2][0] = a.powi(3) * b / 12.0;
        mu[0][2] = a * b.powi(3) / 12.0;
        let eta = normalised(&mu);
        assert!(close(eta[2][0], 1.0 / 6.0, 1e-12), "{a} × {b}: eta20 = {}, expected 1/6", eta[2][0]);
        assert!(close(eta[0][2], 1.0 / 24.0, 1e-12), "{a} × {b}: eta02 = {}, expected 1/24", eta[0][2]);
        assert!(close(eta[0][0], 1.0, 1e-12), "eta00 = {}, expected 1", eta[0][0]);
    }
}

#[test]
fn c3_complex_moments_b3() {
    // unnormalised c20 = 14i, c11 = 28, c30 = 75 + 15i; normalised by mu00^(1 + (p+q)/2), mu00 = 3
    let img = b3();
    let c20 = complex_moment(&img, 2, 0);
    let c11 = complex_moment(&img, 1, 1);
    let c30 = complex_moment(&img, 3, 0);
    let (n2, n3) = (9.0, 3f64.powf(2.5));
    let got = [c20.re as f64 * n2, c20.im as f64 * n2, c11.re as f64 * n2, c30.re as f64 * n3, c30.im as f64 * n3];
    let expected = [0.0, 14.0, 28.0, 75.0, 15.0];
    for k in 0..5 {
        assert!(close(got[k], expected[k], 1e-3), "times mu00^(1+(p+q)/2): c20 = {} + {}i, c11 = {}, c30 = {} + {}i; expected c20 = 14i, c11 = 28, c30 = 75 + 15i", got[0], got[1], got[2], got[3], got[4]);
    }
}

#[test]
fn c3_rotation_turns_the_phase() {
    // rotating the image by 90° counter-clockwise (on the screen) multiplies c_pq by i^(p - q)
    let img = ell();
    let rot = Image::from_fn(40, 40, |x, y| img[(39 - y, x)]);
    for (p, q) in [(2, 0), (3, 0), (2, 1), (1, 2)] {
        let (a, b) = (complex_moment(&img, p, q), complex_moment(&rot, p, q));
        let mut expected = a;
        for _ in 0..(p + 4 - q) % 4 {
            expected = expected * imgcourse::Complex::new(0.0, 1.0);
        }
        assert!((b - expected).abs() <= 1e-5 + 1e-4 * a.abs(), "c{p}{q}: before {a}, after a 90° turn {b}, expected {expected}");
    }
}

#[test]
fn c4_hu_b3_and_mirror() {
    // c30 c12³ = 183180 - 493920i unnormalised; normalised by 3^(2.5 * 4) = 3^10
    let phi = hu(&b3());
    assert!(close(phi[4] * 3f64.powi(10), 183180.0, 1.0), "phi5 · 3^10: got {}, expected 183180", phi[4] * 3f64.powi(10));
    assert!(close(phi[6] * 3f64.powi(10), -493920.0, 2.0), "phi7 · 3^10: got {}, expected -493920", phi[6] * 3f64.powi(10));
    let img = ell();
    let mirror = Image::from_fn(40, 40, |x, y| img[(x, 39 - y)]);
    let (a, b) = (hu(&img), hu(&mirror));
    for k in 0..6 {
        assert!(close(a[k], b[k], 1e-5 * a[k].abs() + 1e-12), "mirror: phi{} changed from {:e} to {:e}", k + 1, a[k], b[k]);
    }
    assert!(close(b[6], -a[6], 1e-5 * a[6].abs()), "mirror: phi7 should change sign, got {:e} and {:e}", a[6], b[6]);
}

#[test]
fn c4_hu_rotation_and_scale() {
    let img = ell();
    let rot = Image::from_fn(40, 40, |x, y| img[(39 - y, x)]);
    let big = Image::from_fn(80, 80, |x, y| img[(x / 2, y / 2)]);
    let (a, r, s) = (hu(&img), hu(&rot), hu(&big));
    for k in 0..7 {
        assert!(close(a[k], r[k], 1e-4 * a[k].abs() + 1e-12), "rotated by 90°: phi{} changed from {:e} to {:e}", k + 1, a[k], r[k]);
        assert!(close(a[k], s[k], 0.01 * a[k].abs() + 1e-12), "twice as large: phi{} changed from {:e} to {:e} (more than 1 %)", k + 1, a[k], s[k]);
    }
}
