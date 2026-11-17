// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p inpainting

use imgcourse::{samples, Image};
use inpainting::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// B1: known 2 at pixel 0 and 10 at pixel 4, pixels 1-3 missing (and 0).
fn b1() -> (Image<f32>, Image<bool>) {
    let img = Image::from_rows(&[[2.0, 0.0, 0.0, 0.0, 10.0]]);
    let missing = Image::from_rows(&[[false, true, true, true, false]]);
    (img, missing)
}

fn assert_row(img: &Image<f32>, expected: &[f32], what: &str) {
    let got = img.row(0);
    let ok = got.iter().zip(expected).all(|(g, e)| close(*g, *e, 1e-4));
    assert!(ok, "{what}: got {got:?}, expected {expected:?}");
}

#[test]
fn c1_jacobi_b1() {
    let (img, missing) = b1();
    let one = jacobi_sweep(&img, &missing);
    assert_row(&one, &[2.0, 1.0, 0.0, 5.0, 10.0], "B1 (b), first Jacobi sweep");
    assert_row(&jacobi_sweep(&one, &missing), &[2.0, 1.0, 3.0, 5.0, 10.0], "B1 (b), second Jacobi sweep");
}

#[test]
fn c2_gauss_seidel_b1() {
    let (mut img, missing) = b1();
    gauss_seidel_sweep(&mut img, &missing);
    assert_row(&img, &[2.0, 1.0, 0.5, 5.25, 10.0], "B1 (c), first Gauss–Seidel sweep");
    gauss_seidel_sweep(&mut img, &missing);
    assert_row(&img, &[2.0, 1.25, 3.25, 6.625, 10.0], "B1 (c), second Gauss–Seidel sweep");
}

#[test]
fn c3_harmonic_b1_b2() {
    let (img, missing) = b1();
    let (u, _) = harmonic_inpaint(&img, &missing, 1e-6);
    let ok = [4.0, 6.0, 8.0].iter().enumerate().all(|(i, e)| close(u[(i + 1, 0)], *e, 1e-3));
    assert!(ok, "B1 (a): got {:?}, expected 2 4 6 8 10", u.row(0));
    // B2: a at (1, 1), b at (2, 1); a's other neighbours are 10, b's are 30.
    let img = Image::from_rows(&[[0.0, 10.0, 30.0, 0.0], [10.0, 0.0, 0.0, 30.0], [0.0, 10.0, 30.0, 0.0]]);
    let mut missing = Image::new(4, 3, false);
    missing[(1, 1)] = true;
    missing[(2, 1)] = true;
    let (u, _) = harmonic_inpaint(&img, &missing, 1e-6);
    assert!(close(u[(1, 1)], 14.0, 1e-3) && close(u[(2, 1)], 26.0, 1e-3), "B2: got a = {}, b = {}, expected 14 and 26", u[(1, 1)], u[(2, 1)]);
}

#[test]
fn c3_known_pixels_untouched() {
    let img = samples::cell().crop(40, 40, 20, 20);
    let missing = Image::from_fn(20, 20, |x, y| (5..12).contains(&x) && (6..10).contains(&y));
    let (u, _) = harmonic_inpaint(&img, &missing, 1e-5);
    for (x, y) in img.coords() {
        if !missing[(x, y)] {
            assert_eq!(u[(x, y)], img[(x, y)], "known pixel ({x}, {y}) was changed");
        }
    }
}

/// B4: the target patch and the candidates A, B, C side by side in one 12 × 3 image.
fn b4() -> (Image<f32>, Image<bool>) {
    let img = Image::from_rows(&[
        [10.0, 12.0, 0.0, 10.0, 11.0, 50.0, 30.0, 30.0, 30.0, 10.0, 12.0, 10.0],
        [11.0, 48.0, 0.0, 12.0, 50.0, 52.0, 30.0, 30.0, 30.0, 11.0, 48.0, 10.0],
        [0.0, 0.0, 52.0, 49.0, 51.0, 50.0, 30.0, 30.0, 30.0, 10.0, 10.0, 10.0],
    ]);
    let missing = Image::from_fn(12, 3, |x, y| (x == 2 && y < 2) || (y == 2 && x < 2));
    (img, missing)
}

#[test]
fn c4_masked_ssd_b4() {
    let (img, missing) = b4();
    for (name, centre, expected) in [("A", (4, 1), 10.0), ("B", (7, 1), 1893.0), ("C", (10, 1), 1764.0)] {
        let d = masked_ssd(&img, &missing, (1, 1), centre, 1);
        assert!(close(d, expected, 1e-3), "B4 (a), candidate {name}: got {d}, expected {expected}");
    }
}

#[test]
fn c5_copy_b4() {
    let (mut img, mut missing) = b4();
    let src = fill_best_patch(&mut img, &mut missing, (1, 1), 1, 20);
    assert_eq!(src, (4, 1), "B4 (b): candidate A, centred at (4, 1), should be copied");
    let got = [img[(2, 0)], img[(2, 1)], img[(0, 2)], img[(1, 2)]];
    assert_eq!(got, [50.0, 52.0, 49.0, 51.0], "B4 (b): the four filled pixels (top right, middle right, bottom left, bottom middle)");
    assert!(missing.pixels().iter().all(|m| !m), "the filled pixels must be marked as known");
}
