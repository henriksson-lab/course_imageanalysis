// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p spline-images

use imgcourse::{samples, Border, Image};
use spline_images::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// A row of 41 pixels with a single 1 in the middle (k = 0 of the exercise is index 20).
fn impulse_row() -> Vec<f32> {
    let mut f = vec![0.0; 41];
    f[20] = 1.0;
    f
}

#[test]
fn c1_impulse_coefficients_b1() {
    let c = prefilter_1d(&impulse_row());
    for (k, expected) in [(0, 1.732), (1, -0.464), (2, 0.124), (3, -0.0333)] {
        assert!(close(c[20 + k], expected, 1e-3), "c_{k}: got {}, expected {expected} (B1b)", c[20 + k]);
        assert!(close(c[20 - k], expected, 1e-3), "c_-{k}: got {}, expected {expected} (symmetric)", c[20 - k]);
    }
}

#[test]
fn c1_prefilter_solves_the_system() {
    // (c[k−1] + 4 c[k] + c[k+1]) / 6 = f[k] at every k, including both ends (mirror outside).
    let f: Vec<f32> = samples::cell().row(100)[60..90].to_vec();
    let c = prefilter_1d(&f);
    let n = f.len();
    for k in 0..n {
        let left = c[Border::Mirror.index(k as isize - 1, n).unwrap()];
        let right = c[Border::Mirror.index(k as isize + 1, n).unwrap()];
        let back = (left + 4.0 * c[k] + right) / 6.0;
        assert!(close(back, f[k], 1e-4), "position {k}: (c[k−1] + 4c[k] + c[k+1])/6 = {back}, but f[k] = {}", f[k]);
    }
}

#[test]
fn c2_c3_interpolates_the_image() {
    let img = samples::cell().crop(80, 90, 12, 9);
    let coef = prefilter(&img);
    for (x, y) in img.coords() {
        let v = bspline_value(&coef, x as f32, y as f32);
        assert!(close(v, img[(x, y)], 1e-4), "pixel ({x}, {y}): the spline gives {v}, the image has {}", img[(x, y)]);
    }
}

#[test]
fn c3_impulse_values_b1() {
    // One row (height 1): the column direction then has all coefficients equal (mirror),
    // so the image behaves like the 1D signal.
    let row = Image::from_vec(41, 1, impulse_row());
    let blurred = bspline_value(&row, 20.0, 0.0); // no prefilter: the pixels as coefficients
    assert!(close(blurred, 2.0 / 3.0, 1e-4), "without prefilter at the impulse: got {blurred}, expected 2/3 (B1a)");
    let half_blur = bspline_value(&row, 20.5, 0.0);
    assert!(close(half_blur, 23.0 / 48.0, 1e-4), "without prefilter halfway: got {half_blur}, expected 23/48 (B1a)");
    let coef = prefilter(&row);
    let at = bspline_value(&coef, 20.0, 0.0);
    let half = bspline_value(&coef, 20.5, 0.0);
    assert!(close(at, 1.0, 1e-4), "with prefilter at the impulse: got {at}, expected 1");
    assert!(close(half, 0.600, 2e-3), "with prefilter halfway: got {half}, expected 0.600 (B1b)");
}

#[test]
fn c4_cubic_convolution_b1_b2() {
    for (s, expected) in [(0.0, 1.0), (0.5, 0.5625), (1.0, 0.0), (1.5, -0.0625), (2.0, 0.0), (-0.5, 0.5625)] {
        let k = cubic_convolution_kernel(s);
        assert!(close(k, expected, 1e-5), "kernel at {s}: got {k}, expected {expected}");
    }
    let row = Image::from_vec(41, 1, impulse_row());
    let half = cubic_convolution(&row, 20.5, 0.0);
    assert!(close(half, 0.5625, 1e-4), "impulse halfway: got {half}, expected 0.5625 (B1c)");
    // B2: a long step 0, 0, 100, 100 with the jump between k = 1 and k = 2 (here 20 and 21).
    let step = Image::from_fn(40, 1, |x, _| if x <= 20 { 0.0 } else { 100.0 });
    for (x, expected) in [(20.5, 50.0), (19.5, -6.25), (21.5, 106.25)] {
        let v = cubic_convolution(&step, x, 0.0);
        assert!(close(v, expected, 1e-3), "step at {x}: got {v}, expected {expected} (B2)");
    }
}

#[test]
fn c5_ffd_b3() {
    // Spacing 32, a 9 × 9 control grid over 0..256; control point (4, 4) at pixel (128, 128).
    let mut d = Image::new(9, 9, [0.0f32, 0.0]);
    d[(4, 4)] = [9.0, 0.0];
    for (x, y, expected) in [(128.0, 128.0, 4.0), (160.0, 128.0, 1.0), (160.0, 160.0, 0.25), (192.0, 128.0, 0.0), (144.0, 128.0, 2.875)] {
        let u = ffd_displacement(&d, 32.0, x, y);
        assert!(close(u[0], expected, 1e-4) && close(u[1], 0.0, 1e-6), "pixel ({x}, {y}) moves by {u:?}, expected ({expected}, 0) (B3)");
    }
    // All control points move by the same vector: a translation (B3c).
    let all = Image::new(9, 9, [9.0f32, -2.0]);
    for (x, y) in [(100.0, 90.0), (50.5, 200.25), (128.0, 32.0)] {
        let u = ffd_displacement(&all, 32.0, x, y);
        assert!(close(u[0], 9.0, 1e-4) && close(u[1], -2.0, 1e-4), "translation: pixel ({x}, {y}) moves by {u:?}, expected (9, -2)");
    }
}
