// Checks for the coding exercises, using the numbers of the by-hand exercises.
// Run: cargo test -p imageformation illposed::

use imgcourse::{samples, Image};
use imageformation::illposed::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn check_row(what: &str, got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len(), "{what}: length {}, expected {}", got.len(), expected.len());
    for (g, e) in got.iter().zip(expected) {
        assert!(close(*g, *e, 1e-5), "{what}: got {got:?}, expected {expected:?}");
    }
}

fn gaussian_psf(sigma: f32) -> Image<f32> {
    let r = (3.0 * sigma).ceil() as usize;
    let g = Image::from_fn(2 * r + 1, 2 * r + 1, |x, y| {
        let (dx, dy) = (x as f32 - r as f32, y as f32 - r as f32);
        (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp()
    });
    let total = g.sum();
    g.map(|v| v / total)
}

const K121: [f32; 3] = [0.25, 0.5, 0.25];

#[test]
fn c1_b2_same_image() {
    let expected = [2.25, 1.75, 1.75, 2.25];
    check_row("blur of (3, 1, 2, 2)", &circular_blur(&[3., 1., 2., 2.], &K121), &expected);
    check_row("blur of (4, 0, 3, 1)", &circular_blur(&[4., 0., 3., 1.], &K121), &expected);
    check_row("blur of (3.5, 0.5, 2.5, 1.5)", &circular_blur(&[3.5, 0.5, 2.5, 1.5], &K121), &expected);
    check_row("blur of the fastest wave (1, −1, 1, −1)", &circular_blur(&[1., -1., 1., -1.], &K121), &[0.; 4]);
}

#[test]
fn c1_wraps_around() {
    // An asymmetric kernel shows which neighbour is which: (1, 0, 0) takes the left neighbour.
    check_row("kernel (1, 0, 0) on (1, 2, 3, 4)", &circular_blur(&[1., 2., 3., 4.], &[1., 0., 0.]), &[4., 1., 2., 3.]);
}

#[test]
fn c2_transfer() {
    check_row("H of (1, 2, 1)/4 on 4 pixels", &transfer(&K121, 4), &[1.0, 0.5, 0.0, 0.5]);
    check_row("H of (0.2, 0.6, 0.2) on 4 pixels (B7)", &transfer(&[0.2, 0.6, 0.2], 4), &[1.0, 0.6, 0.2, 0.6]);
}

#[test]
fn c2_eigenvalues() {
    // H(k) is the eigenvalue of the wave cos(2π k j / n): blurring the wave multiplies it by H(k).
    let (n, kernel) = (8, [0.1, 0.2, 0.4, 0.2, 0.1]);
    let h = transfer(&kernel, n);
    for k in 0..n {
        let wave: Vec<f32> = (0..n).map(|j| (2.0 * std::f32::consts::PI * (k * j) as f32 / n as f32).cos()).collect();
        let expected: Vec<f32> = wave.iter().map(|v| h[k] * v).collect();
        check_row(&format!("blurred wave k = {k}, H(k) = {}", h[k]), &circular_blur(&wave, &kernel), &expected);
    }
}

#[test]
fn c3_condition_number() {
    let k = condition_number(&[0.2, 0.6, 0.2], 4);
    assert!(close(k, 5.0, 1e-4), "B7: got {k}, expected 5");
    let k = condition_number(&K121, 4);
    assert!(k.is_infinite(), "(1, 2, 1)/4 on 4 pixels: got {k}, expected infinite (H = 0 for the fastest wave)");
    let k = condition_number(&K121, 5);
    assert!(k.is_finite() && k > 5.0, "(1, 2, 1)/4 on 5 pixels: got {k}, expected finite (an odd row has no wave (1, −1, 1, −1))");
}

#[test]
fn c4_otf_of_a_gaussian() {
    let sigma = 1.5;
    let h = otf2(&gaussian_psf(sigma), 32, 16);
    assert_eq!((h.width(), h.height()), (32, 16), "size: got {} × {}", h.width(), h.height());
    assert!(close(h[(0, 0)].re, 1.0, 1e-4), "H at frequency 0: got {}, expected 1 (the PSF sums to 1)", h[(0, 0)]);
    for (x, y) in h.coords() {
        assert!(h[(x, y)].im.abs() < 1e-4, "frequency ({x}, {y}): H = {}, expected real (the PSF is symmetric and centred on (0, 0))", h[(x, y)]);
    }
    // 3 cycles along x on 32 pixels: f = 3/32 cycles/pixel, H = exp(−2π²σ²f²)
    let f = 3.0 / 32.0;
    let e = (-2.0 * std::f32::consts::PI.powi(2) * sigma * sigma * f * f).exp();
    assert!(close(h[(3, 0)].re, e, 2e-3), "H at 3 cycles along x: got {}, expected about {e}", h[(3, 0)].re);
}

#[test]
fn c4_point_comes_out_as_the_psf() {
    // An asymmetric PSF, blurred onto a single bright pixel at (5, 4): the PSF must appear
    // around it, the right way round.
    let psf = Image::from_rows(&[[0.0, 0.1, 0.0], [0.0, 0.4, 0.3], [0.0, 0.2, 0.0]]);
    let mut point = Image::new(12, 10, 0.0);
    point[(5, 4)] = 1.0;
    let out = blur_periodic(&point, &otf2(&psf, 12, 10));
    for (i, j) in psf.coords() {
        let (x, y) = (5 + i - 1, 4 + j - 1);
        assert!(close(out[(x, y)], psf[(i, j)], 1e-5), "pixel ({x}, {y}): got {}, expected {}", out[(x, y)], psf[(i, j)]);
    }
}

#[test]
fn c5_inverse_without_noise_is_exact() {
    let f = samples::cell().crop(90, 70, 32, 32);
    let otf = otf2(&gaussian_psf(0.7), 32, 32);
    let g = blur_periodic(&f, &otf);
    let back = inverse_filter(&g, &otf);
    for (x, y) in f.coords() {
        assert!(close(back[(x, y)], f[(x, y)], 2e-3), "pixel ({x}, {y}): restored {}, object {}", back[(x, y)], f[(x, y)]);
    }
}

#[test]
fn c5_invisible_checkerboard() {
    // The blur (1, 2, 1)/4 along x and along y kills the checkerboard: object and
    // object + checkerboard give the same image, and the inverse filter cannot tell them apart.
    let psf = Image::from_rows(&[[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]).map(|v| v / 16.0);
    let f = samples::cell().crop(30, 40, 16, 16);
    let f2 = f.map_xy(|x, y, v| v + if (x + y) % 2 == 0 { 0.2 } else { -0.2 });
    let otf = otf2(&psf, 16, 16);
    let (g, g2) = (blur_periodic(&f, &otf), blur_periodic(&f2, &otf));
    let a = inverse_filter(&g, &otf);
    let b = inverse_filter(&g2, &otf);
    for (x, y) in f.coords() {
        assert!(close(g[(x, y)], g2[(x, y)], 1e-5), "pixel ({x}, {y}): the two images differ, {} and {}", g[(x, y)], g2[(x, y)]);
        assert!(a[(x, y)].is_finite(), "pixel ({x}, {y}): the inverse filter gave {} (divided by zero?)", a[(x, y)]);
        assert!(close(a[(x, y)], b[(x, y)], 1e-3), "pixel ({x}, {y}): restored {} and {} from the same data", a[(x, y)], b[(x, y)]);
    }
}
