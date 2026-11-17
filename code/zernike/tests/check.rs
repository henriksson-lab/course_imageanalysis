// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p zernike

use imgcourse::{samples, Complex, Image};
use zernike::*;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn near(a: Complex, b: Complex, tol: f32) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn c1_radial_small_ones() {
    // R20 = 2ρ² - 1, R22 = ρ², R31 = 3ρ³ - 2ρ, R33 = ρ³, R40 = 6ρ⁴ - 6ρ² + 1, at ρ = 0.5
    let cases = [(2, 0, -0.5), (2, 2, 0.25), (3, 1, -0.625), (3, 3, 0.125), (4, 0, -0.125), (1, -1, 0.5)];
    for (n, m, expected) in cases {
        let got = radial(n, m, 0.5);
        assert!(close(got, expected, 1e-12), "R{n},{m}(0.5): got {got}, expected {expected}");
    }
    for (n, m) in [(6, 2), (8, 0), (10, 4), (9, 3)] {
        assert!(close(radial(n, m, 1.0), 1.0, 1e-9), "R{n},{m}(1): got {}, expected 1", radial(n, m, 1.0));
    }
}

#[test]
fn c1_radial_orthogonal_b1() {
    // midpoint sums of the integrals of B1 over [0, 1] with the weight ρ
    let steps = 10000;
    let integral = |f: &dyn Fn(f64) -> f64| (0..steps).map(|i| f((i as f64 + 0.5) / steps as f64) / steps as f64).sum::<f64>();
    let a = integral(&|r| radial(2, 0, r) * radial(0, 0, r) * r);
    let b = integral(&|r| radial(3, 1, r) * radial(1, 1, r) * r);
    let c = integral(&|r| radial(2, 0, r).powi(2) * r);
    assert!(close(a, 0.0, 1e-6), "∫ R20 R00 ρ dρ: got {a:.2e}, expected 0");
    assert!(close(b, 0.0, 1e-6), "∫ R31 R11 ρ dρ: got {b:.2e}, expected 0");
    assert!(close(c, 1.0 / 6.0, 1e-6), "∫ R20² ρ dρ: got {c:.6}, expected 1/6");
}

#[test]
fn c2_constant_image() {
    // I = 1 on the disc: Z00 = (1/π) · area of the disc = 1 (up to the pixel staircase), the others 0
    let img = Image::new(64, 64, 1.0);
    let z00 = zernike_moment(&img, 0, 0);
    assert!(close(z00.re as f64, 1.0, 0.01) && z00.im.abs() < 1e-6, "Z00 of a constant image: got {z00}, expected about 1");
    // (m = 4 is left out: the pixel staircase of the disc has the 4-fold symmetry of the grid)
    for (n, m) in [(1, 1), (2, 2), (3, 1), (4, 2)] {
        let z = zernike_moment(&img, n, m);
        assert!(z.abs() < 1e-4, "Z{n}{m} of a constant image: got {z}, expected 0 (the disc is symmetric)");
    }
}

#[test]
fn c2_rotation_and_mirror_b2() {
    let img = samples::letter_f(64);
    // a 90° turn counter-clockwise on the screen: Z_nm -> Z_nm e^(-i m 90°) = Z_nm (-i)^m
    let rot = Image::from_fn(64, 64, |x, y| img[(63 - y, x)]);
    let top_bottom = Image::from_fn(64, 64, |x, y| img[(x, 63 - y)]);
    let left_right = Image::from_fn(64, 64, |x, y| img[(63 - x, y)]);
    for (n, m) in [(1, 1), (2, 2), (3, 1), (3, 3), (4, 2)] {
        let z = zernike_moment(&img, n, m);
        let mut expected = z;
        for _ in 0..m {
            expected = expected * Complex::new(0.0, -1.0);
        }
        let got = zernike_moment(&rot, n, m);
        assert!(near(got, expected, 1e-4), "Z{n}{m}: {z} before, {got} after a 90° turn, expected {expected}");
        let got = zernike_moment(&top_bottom, n, m);
        assert!(near(got, z.conj(), 1e-4), "Z{n}{m}: {z} before, {got} after a top–bottom mirror, expected the conjugate");
        let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
        let got = zernike_moment(&left_right, n, m);
        assert!(near(got, z.conj() * sign, 1e-4), "Z{n}{m}: {z} before, {got} after a left–right mirror, expected (-1)^m times the conjugate (B11)");
        let neg = zernike_moment(&img, n, -m);
        assert!(near(neg, z.conj(), 1e-4), "Z{n},-{m} = {neg}: expected the conjugate of Z{n}{m} = {z}");
    }
}

#[test]
fn c3_reconstruct_a_low_order_image() {
    // 0.5 + 0.3 V20 + 0.2 Re V22 on the disc comes back from the moments up to order 2
    let size = 96;
    let img = Image::from_fn(size, size, |x, y| {
        let (rho, theta) = polar(x, y, size);
        if rho <= 1.0 { (0.5 + 0.3 * (2.0 * rho * rho - 1.0) + 0.2 * rho * rho * (2.0 * theta).cos()) as f32 } else { 0.0 }
    });
    let rec = reconstruct(&img, 2);
    let mut worst = 0.0f32;
    for (x, y) in img.coords() {
        if polar(x, y, size).0 <= 0.9 {
            worst = worst.max((rec[(x, y)] - img[(x, y)]).abs());
        }
    }
    assert!(worst < 0.01, "inside ρ = 0.9 the reconstruction should match: largest difference {worst}");
    assert_eq!(rec[(0, 0)], 0.0, "the corner lies outside the disc: expected 0");
}

#[test]
fn c4_recurrence_equals_formula() {
    for n in (0..=20).step_by(2) {
        for rho in [0.0, 0.3, 0.5, 0.71, 0.9, 1.0] {
            let (a, b) = (radial0_recurrence(n, rho), radial(n, 0, rho));
            assert!(close(a, b, 1e-8), "R{n},0({rho}): recurrence {a}, factorial formula {b}");
        }
    }
}
