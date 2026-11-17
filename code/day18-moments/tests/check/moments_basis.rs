// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day18 moments_basis::

use imgcourse::{samples, Image};
use day18::moments_basis::*;

/// The 4 × 4 grey image of B2 (rows from the top).
fn b2() -> Image<f32> {
    Image::from_rows(&[[0., 1., 2., 1.], [0., 2., 3., 1.], [0., 1., 2., 0.], [0., 0., 1., 0.]])
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn c1_raw_moments_b2() {
    let m = raw_moments(&b2(), 2);
    let got = [m[0][0], m[1][0], m[0][1], m[2][0], m[1][1], m[0][2]];
    let expected = [14.0, 26.0, 15.0, 54.0, 27.0, 27.0];
    for (k, name) in ["m00", "m10", "m01", "m20", "m11", "m02"].iter().enumerate() {
        assert!(close(got[k], expected[k], 1e-9), "{name}: got {}, expected {}", got[k], expected[k]);
    }
}

#[test]
fn c2_central_moments_b2() {
    let mu = central_moments(&b2(), 3);
    let cases = [("mu20", mu[2][0], 5.714), ("mu11", mu[1][1], -0.857), ("mu02", mu[0][2], 10.929), ("mu30", mu[3][0], 0.490), ("mu03", mu[0][3], 4.653)];
    for (name, got, expected) in cases {
        assert!(close(got, expected, 1e-3), "{name}: got {got:.4}, expected {expected}");
    }
    assert!(close(mu[1][0], 0.0, 1e-9) && close(mu[0][1], 0.0, 1e-9), "mu10 and mu01 must be 0: got {} and {}", mu[1][0], mu[0][1]);
}

#[test]
fn c2_central_moments_ignore_a_shift() {
    let img = b2();
    let shifted = Image::from_fn(9, 7, |x, y| if (3..7).contains(&x) && (2..6).contains(&y) { img[(x - 3, y - 2)] } else { 0.0 });
    let (a, b) = (central_moments(&img, 3), central_moments(&shifted, 3));
    for p in 0..=3 {
        for q in 0..=3 - p {
            assert!(close(a[p][q], b[p][q], 1e-9), "mu{p}{q}: {} before the shift, {} after", a[p][q], b[p][q]);
        }
    }
}

#[test]
fn c3_skewness_b2() {
    let (gx, kx, gy, _) = skewness_kurtosis(&central_moments(&b2(), 4));
    assert!(close(gx, 0.134, 1e-3), "skewness along x: got {gx:.4}, expected 0.134");
    assert!(close(gy, 0.482, 1e-3), "skewness along y: got {gy:.4}, expected 0.482");
    // the mirror image (left-right) has the opposite skewness along x, the same kurtosis
    let mirror = Image::from_fn(4, 4, |x, y| b2()[(3 - x, y)]);
    let (mx, mk, _, _) = skewness_kurtosis(&central_moments(&mirror, 4));
    assert!(close(mx, -gx, 1e-9), "mirrored: skewness {mx:.4}, expected {:.4}", -gx);
    assert!(close(mk, kx, 1e-9), "mirrored: kurtosis {mk:.4}, expected {kx:.4} (unchanged)");
}

#[test]
fn c3_kurtosis_square_and_disc_b5() {
    let square = Image::from_fn(40, 40, |x, y| if (5..35).contains(&x) && (5..35).contains(&y) { 1.0 } else { 0.0 });
    let r = 30.0 / 3f32.sqrt();
    let disc = Image::from_fn(40, 40, |x, y| if (x as f32 - 19.5).hypot(y as f32 - 19.5) <= r { 1.0 } else { 0.0 });
    let (_, ks, _, _) = skewness_kurtosis(&central_moments(&square, 4));
    let (_, kd, _, _) = skewness_kurtosis(&central_moments(&disc, 4));
    assert!(close(ks, 1.797, 2e-3), "square of side 30: kurtosis {ks:.4}, expected 1.797 (1.8 for the continuous square)");
    assert!(close(kd, 2.0, 0.02), "disc of radius 17.3: kurtosis {kd:.4}, expected about 2.0");
}

#[test]
fn c4_separable_equals_direct() {
    let img = samples::cell().crop(60, 80, 40, 30);
    let (a, b) = (raw_moments(&img, 4), moments_separable(&img, 4, false));
    for p in 0..=4 {
        for q in 0..=4 - p {
            assert!(close(a[p][q], b[p][q], 1e-9 * a[p][q].abs().max(1.0)), "m{p}{q}: separable {} but direct {}", b[p][q], a[p][q]);
        }
    }
}

#[test]
fn c4_scaled_coordinates() {
    // 4 × 4 ones: the centres become -3/4, -1/4, 1/4, 3/4 in both directions
    let m = moments_separable(&Image::new(4, 4, 1.0), 4, true);
    let cases = [("m00", m[0][0], 16.0), ("m10", m[1][0], 0.0), ("m20", m[2][0], 5.0), ("m11", m[1][1], 0.0), ("m40", m[4][0], 2.5625)];
    for (name, got, expected) in cases {
        assert!(close(got, expected, 1e-9), "{name} in [-1, 1]: got {got}, expected {expected}");
    }
}
