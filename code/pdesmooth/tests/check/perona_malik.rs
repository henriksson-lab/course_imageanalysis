// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p pdesmooth perona_malik::

use imgcourse::{samples, Image};
use pdesmooth::perona_malik::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn assert_row(got: &[f32], expected: &[f32], tol: f32, what: &str) {
    let ok = got.len() == expected.len() && got.iter().zip(expected).all(|(g, e)| close(*g, *e, tol));
    assert!(ok, "{what}: got {got:?}, expected {expected:?}");
}

#[test]
fn c1_conductivity_b1() {
    for (s, expected) in [(1.0, 1.0 / 1.04), (5.0, 0.5), (20.0, 1.0 / 17.0)] {
        let got = g2(s, 5.0);
        assert!(close(got, expected, 1e-4), "g2({s}) with K = 5: got {got}, expected {expected:.4}");
    }
    let got = g1(5.0, 5.0);
    assert!(close(got, (-1.0f32).exp(), 1e-4), "g1(K) must be e^-1 = 0.368, got {got}");
    assert!(close(g1(0.0, 5.0), 1.0, 1e-6) && close(g2(0.0, 5.0), 1.0, 1e-6), "g(0) must be 1");
}

#[test]
fn c2_one_step_b2() {
    let row = [10.0, 12.0, 10.0, 50.0, 50.0];
    let pm = pm_step_1d(&row, 0.25, 5.0, g2);
    assert_row(&pm, &[10.431, 11.138, 10.585, 49.846, 50.0], 2e-3, "Perona–Malik step, g2, K = 5");
    // With a huge K, g is 1 everywhere: the heat step of B2 (a).
    let heat = pm_step_1d(&row, 0.25, 1e9, g2);
    assert_row(&heat, &[10.5, 11.0, 20.5, 40.0, 50.0], 1e-3, "K very large (the heat step)");
}

#[test]
fn c2_sharpening_b3() {
    let pm = pm_step_1d(&[0.0, 2.0, 6.0, 10.0, 12.0], 0.25, 2.0, g2);
    assert_row(&pm, &[0.25, 1.95, 6.0, 10.05, 11.75], 1e-4, "B3, g2, K = 2");
}

#[test]
fn c2_total_kept() {
    let row = [3.0, 1.0, 40.0, 1.0, 5.0, 9.0, 2.0];
    let total: f32 = pm_step_1d(&row, 0.4, 3.0, g1).iter().sum();
    assert!(close(total, 61.0, 1e-3), "the total must stay 61, got {total}");
}

fn spike() -> Image<f32> {
    let mut img = Image::new(9, 9, 0.0);
    img[(4, 4)] = 30.0;
    img
}

#[test]
fn c3_spike_b4() {
    let a = pm_step(&spike(), 0.25, 5.0, g2, 0.0)[(4, 4)];
    assert!(close(a, 29.19, 0.01), "B4 (a), g2: got {a}, expected 29.19");
    let b = pm_step(&spike(), 0.25, 5.0, g1, 0.0)[(4, 4)];
    assert!(close(b, 30.0, 1e-3), "B4 (b), g1: the spike is frozen, got {b}, expected 30");
    let c = pm_step(&spike(), 0.25, 5.0, g1, 1.0)[(4, 4)];
    assert!(c < 10.0, "B4 (c), g1 with pre-smoothing sigma = 1: got {c}, expected far below 30 (about 4)");
}

#[test]
fn c3_huge_k_is_heat() {
    let img = samples::cell().crop(10, 10, 20, 20);
    let pm = pm_step(&img, 0.2, 1e9, g2, 0.0);
    for (x, y) in [(0, 0), (5, 7), (19, 19)] {
        let (xi, yi) = (x as isize, y as isize);
        let b = imgcourse::Border::Mirror;
        let n = img.get_border(xi - 1, yi, b) + img.get_border(xi + 1, yi, b) + img.get_border(xi, yi - 1, b) + img.get_border(xi, yi + 1, b);
        let heat = img[(x, y)] + 0.2 * (n - 4.0 * img[(x, y)]);
        assert!(close(pm[(x, y)], heat, 1e-5), "K huge, pixel ({x}, {y}): got {}, expected the heat step {heat}", pm[(x, y)]);
    }
}

#[test]
fn c4_psnr() {
    let truth = Image::new(4, 4, 0.5);
    let off = truth.map(|v| v + 0.1); // error 0.1 everywhere: MSE 0.01, PSNR 20 dB
    let p = psnr(&off, &truth, 1.0);
    assert!(close(p, 20.0, 1e-3), "every pixel off by 0.1: got {p} dB, expected 20");
    let p = psnr(&off.map(|v| 255.0 * v), &truth.map(|v| 255.0 * v), 255.0);
    assert!(close(p, 20.0, 1e-2), "the same on the 0-255 scale with peak 255: got {p} dB, expected 20");
}
