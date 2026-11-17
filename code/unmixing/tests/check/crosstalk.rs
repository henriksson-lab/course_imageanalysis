// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p unmixing crosstalk::

use unmixing::crosstalk::*;
use imgcourse::Image;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

const DYE1: [f32; 5] = [0.2, 1.0, 0.6, 0.3, 0.1];
const DYE2: [f32; 5] = [0.0, 0.1, 0.5, 1.0, 0.6];

#[test]
fn c1_mixing_matrix_b1() {
    let f1 = [1.0, 1.0, 0.0, 0.0, 0.0];
    let f2 = [0.0, 0.0, 0.0, 1.0, 1.0];
    let m = mixing_matrix([&DYE1, &DYE2], [&f1, &f2], 25.0);
    let e = [[30.0, 2.5], [10.0, 40.0]];
    for c in 0..2 {
        for k in 0..2 {
            assert!(close(m[c][k], e[c][k], 1e-3), "B1(a), channel {}, dye {}: got {}, expected {}", c + 1, k + 1, m[c][k], e[c][k]);
        }
    }
    // B1(c): channel 2 also takes half of the 550-nm bin
    let f2 = [0.0, 0.0, 0.5, 1.0, 1.0];
    let m = mixing_matrix([&DYE1, &DYE2], [&f1, &f2], 25.0);
    assert!(close(m[1][0], 17.5, 1e-3) && close(m[1][1], 46.25, 1e-3), "B1(c): got channel 2 = ({}, {}), expected (17.5, 46.25)", m[1][0], m[1][1]);
}

#[test]
fn c2_slope_b6() {
    let s = slope_through_origin(&[100.0, 200.0, 300.0, 400.0], &[27.0, 48.0, 77.0, 98.0]);
    assert!(close(s, 74600.0 / 300000.0, 1e-5), "B6(a): got {s}, expected 0.249");
}

#[test]
fn c3_inverse_b3() {
    let inv = inverse2([[1.0, 0.1], [0.2, 1.0]]);
    let e = [[1.020, -0.102], [-0.204, 1.020]];
    for r in 0..2 {
        for c in 0..2 {
            assert!(close(inv[r][c], e[r][c], 1e-3), "B3, row {}, column {}: got {}, expected {}", r + 1, c + 1, inv[r][c], e[r][c]);
        }
    }
}

#[test]
fn c3_compensate_b4_b5() {
    let ch1 = Image::from_rows(&[[2100.0f32, 1000.0, 1000.0]]);
    let ch2 = Image::from_rows(&[[1400.0f32, 900.0, 910.0]]);
    let (a1, a2) = compensate(&ch1, &ch2, [[1.0, 0.1], [0.2, 1.0]]);
    assert!(close(a1[(0, 0)], 2000.0, 0.05) && close(a2[(0, 0)], 1000.0, 0.05), "B4: got ({}, {}), expected (2000, 1000)", a1[(0, 0)], a2[(0, 0)]);
    let (a1, a2) = compensate(&ch1, &ch2, [[1.0, 0.9], [0.9, 1.0]]);
    assert!(close(a1[(1, 0)], 1000.0, 0.05) && close(a2[(1, 0)], 0.0, 0.05), "B5, (1000, 900): got ({}, {}), expected (1000, 0)", a1[(1, 0)], a2[(1, 0)]);
    assert!(close(a1[(2, 0)], 952.6, 0.1) && close(a2[(2, 0)], 52.6, 0.1), "B5, (1000, 910): got ({}, {}), expected (952.6, 52.6)", a1[(2, 0)], a2[(2, 0)]);
}

#[test]
fn c3_wrong_guess_b7() {
    // 400 units of dye 1 records (400, 100); compensated dye 2 = 100 − g · 400.
    let ch1 = Image::new(1, 1, 400.0f32);
    let ch2 = Image::new(1, 1, 100.0f32);
    for (g, e) in [(0.15, 40.0), (0.25, 0.0), (0.35, -40.0)] {
        let (_, a2) = compensate(&ch1, &ch2, [[1.0, 0.0], [g, 1.0]]);
        assert!(close(a2[(0, 0)], e, 1e-3), "B7(b), guess {g}: got {}, expected {e}", a2[(0, 0)]);
    }
}

#[test]
fn c4_pearson() {
    let mask = Image::new(4, 1, true);
    let a = Image::from_rows(&[[1.0f32, 2.0, 3.0, 4.0]]);
    let r = pearson(&a, &a.map(|v| 3.0 * v + 5.0), &mask);
    assert!(close(r, 1.0, 1e-5), "a line going up: got r = {r}, expected 1");
    let r = pearson(&a, &a.map(|v| -v), &mask);
    assert!(close(r, -1.0, 1e-5), "a line going down: got r = {r}, expected −1");
    let b = Image::from_rows(&[[1.0f32, -1.0, -1.0, 1.0]]);
    let r = pearson(&a, &b, &mask);
    assert!(close(r, 0.0, 1e-5), "no linear relation: got r = {r}, expected 0");
    let partial = Image::from_rows(&[[true, true, true, false]]);
    let c = Image::from_rows(&[[1.0f32, 2.0, 3.0, -100.0]]);
    let r = pearson(&a, &c, &partial);
    assert!(close(r, 1.0, 1e-5), "only the masked pixels count: got r = {r}, expected 1");
}
