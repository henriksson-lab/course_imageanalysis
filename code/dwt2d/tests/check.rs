// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p dwt2d

use dwt2d::*;
use imgcourse::{samples, Image};

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn assert_image(got: &Image<f32>, expected: &Image<f32>, tol: f32, what: &str) {
    for (x, y) in expected.coords() {
        assert!(close(got[(x, y)], expected[(x, y)], tol), "{what}, position ({x}, {y}): got {}, expected {}\ngot:\n{got:.3}expected:\n{expected:.3}", got[(x, y)], expected[(x, y)]);
    }
}

#[test]
fn c1_two_by_two_blocks_b1() {
    // One level on a 2 × 2 block gives [[LL, V], [H, D]].
    let cases = [
        ([[5.0, 5.0], [5.0, 5.0]], [[10.0, 0.0], [0.0, 0.0]], "(a) flat"),
        ([[2.0, 8.0], [2.0, 8.0]], [[10.0, -6.0], [0.0, 0.0]], "(b) vertical edge"),
        ([[2.0, 2.0], [8.0, 8.0]], [[10.0, 0.0], [-6.0, 0.0]], "(c) horizontal edge"),
        ([[8.0, 2.0], [2.0, 8.0]], [[10.0, 0.0], [0.0, 6.0]], "(d) diagonal"),
    ];
    for (block, bands, name) in cases {
        let got = dwt2d_level(&Image::from_rows(&block), &HAAR);
        assert_image(&got, &Image::from_rows(&bands), 1e-4, &format!("B1{name}: [[LL, V], [H, D]]"));
    }
}

#[test]
fn c1_horizontal_edge_b2a() {
    let img = Image::from_rows(&[[1.0; 4], [1.0; 4], [1.0; 4], [9.0; 4]]);
    let got = dwt2d_level(&img, &HAAR);
    let expected = Image::from_rows(&[[2.0, 2.0, 0.0, 0.0], [10.0, 10.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-8.0, -8.0, 0.0, 0.0]]);
    assert_image(&got, &expected, 1e-4, "B2(a): LL top left, V top right, H bottom left, D bottom right");
}

#[test]
fn c2_inverse_level() {
    let img = samples::cell().crop(40, 60, 16, 12);
    for (name, h) in [("Haar", &HAAR[..]), ("db2", &DB2[..]), ("db4", &DB4[..])] {
        let back = idwt2d_level(&dwt2d_level(&img, h), h);
        assert_image(&back, &img, 1e-4, &format!("{name}: one level and back"));
    }
}

#[test]
fn c3_pyramid_b2b() {
    let img = Image::from_rows(&[[1.0; 4], [1.0; 4], [9.0; 4], [9.0; 4]]);
    let got = dwt2d(&img, &HAAR, 2);
    assert!(close(got[(0, 0)], 20.0, 1e-4), "B2(b), two levels: LL = {}, expected 20", got[(0, 0)]);
    assert!(close(got[(0, 1)], -16.0, 1e-4), "B2(b), two levels: H = {}, expected -16", got[(0, 1)]);
    for (x, y) in got.coords() {
        if (x, y) != (0, 0) && (x, y) != (0, 1) {
            assert!(close(got[(x, y)], 0.0, 1e-4), "B2(b): position ({x}, {y}) is {}, expected 0 (all level-1 details vanish)", got[(x, y)]);
        }
    }
    let img = samples::cell().crop(0, 0, 64, 64);
    let c = dwt2d(&img, &DB4, 3);
    let ex: f32 = img.pixels().iter().map(|v| v * v).sum();
    let ec: f32 = c.pixels().iter().map(|v| v * v).sum();
    assert!(close(ex, ec, 1e-3 * ex), "db4, 3 levels: Σx² = {ex} but Σc² = {ec} (orthogonal: they must agree)");
    assert_image(&idwt2d(&c, &DB4, 3), &img, 1e-4, "db4, 3 levels and back");
}

#[test]
fn c4_keep_and_psnr_b5() {
    // B5: dropped energy 10⁶ on 256 × 256 pixels (grey levels 0..255): MSE 15.26, PSNR 36.3 dB.
    let a = Image::new(256, 256, 100.0f32);
    let b = a.map(|v| v + (1e6f32 / 65536.0).sqrt());
    let p = psnr(&a, &b, 255.0);
    assert!(close(p, 36.29, 0.02), "PSNR for MSE 15.26: got {p}, expected 36.3 dB (B5)");
    let b2 = a.map(|v| v + (2e6f32 / 65536.0).sqrt());
    let p2 = psnr(&a, &b2, 255.0);
    assert!(close(p - p2, 3.01, 0.02), "twice the dropped energy: PSNR {p2}, expected 3 dB less than {p}");
    // In an orthonormal transform the squared error is the dropped energy.
    let img = samples::cell().crop(64, 64, 64, 64);
    let c = dwt2d(&img, &DB2, 3);
    let (kept, dropped) = keep_fraction(&c, 0.1);
    let nonzero = kept.pixels().iter().filter(|v| **v != 0.0).count();
    assert!((405..=415).contains(&nonzero), "keeping 10 % of 4096 coefficients: {nonzero} kept, expected about 410");
    let back = idwt2d(&kept, &DB2, 3);
    let err: f32 = back.pixels().iter().zip(img.pixels()).map(|(p, q)| (p - q) * (p - q)).sum();
    assert!(close(err, dropped, 1e-3 * dropped.max(1e-3)), "squared error {err} should equal the dropped energy {dropped}");
}
