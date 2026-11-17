// Checks for the coding exercises, using the numbers of the lecture and of the by-hand exercises.
// Run: cargo test -p richardson-lucy

use imgcourse::{samples, Image};
use richardson_lucy::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn row(v: [f32; 4]) -> Image<f32> {
    Image::from_rows(&[v])
}

fn psf121() -> Image<f32> {
    Image::from_rows(&[[0.25, 0.5, 0.25]])
}

fn check(what: &str, got: &Image<f32>, expected: &[f32]) {
    for (i, (g, e)) in got.pixels().iter().zip(expected).enumerate() {
        assert!(close(*g, *e, 1e-3), "{what}, pixel {}: got {g}, expected {e} (all: {:?})", i + 1, got.pixels());
    }
}

#[test]
fn c1_blur_wraps() {
    check("blur of (1, 1, 1, 1)", &blur(&row([1., 1., 1., 1.]), &psf121()), &[1., 1., 1., 1.]);
    check("blur of (1, 0, 0, 0): the left neighbour of pixel 1 is pixel 4", &blur(&row([1., 0., 0., 0.]), &psf121()), &[0.5, 0.25, 0., 0.25]);
    // An asymmetric PSF: (1, 0, 0) moves the light one pixel to the left.
    check("blur of (0, 1, 0, 0) with the PSF (1, 0, 0)", &blur(&row([0., 1., 0., 0.]), &Image::from_rows(&[[1., 0., 0.]])), &[1., 0., 0., 0.]);
}

#[test]
fn c2_lecture_table() {
    let y = row([1., 2., 1., 0.]);
    let x1 = rl_step(&row([1., 1., 1., 1.]), &y, &psf121(), 0.0);
    check("x1", &x1, &[1., 1.5, 1., 0.5]);
    let x2 = rl_step(&x1, &y, &psf121(), 0.0);
    check("x2", &x2, &[0.9, 1.95, 0.9, 0.25]);
}

#[test]
fn c2_b1() {
    let y = row([0., 2., 2., 0.]);
    let x1 = rl_step(&row([1., 1., 1., 1.]), &y, &psf121(), 0.0);
    check("B1 x1", &x1, &[0.5, 1.5, 1.5, 0.5]);
    let x2 = rl_step(&x1, &y, &psf121(), 0.0);
    check("B1 x2", &x2, &[0.2, 1.8, 1.8, 0.2]);
}

#[test]
fn c2_spreads_back_with_the_mirrored_psf() {
    // The PSF (0, 0, 1) moves the light one pixel to the right. The data are the object
    // (0, 3, 1, 0) moved right; one step from flat must move the ratio back to the left.
    let psf = Image::from_rows(&[[0., 0., 1.]]);
    let x1 = rl_step(&row([1., 1., 1., 1.]), &row([0., 0., 3., 1.]), &psf, 0.0);
    check("one step with a shifting PSF", &x1, &[0., 3., 1., 0.]);
}

#[test]
fn c3_b1_third_step() {
    let x3 = richardson_lucy(&row([0., 2., 2., 0.]), &psf121(), 3, 0.0);
    check("B1 x3 (flat start = the mean)", &x3, &[0.071, 1.929, 1.929, 0.071]);
}

#[test]
fn c3_keeps_the_photons_and_stays_positive() {
    let y = samples::cell().crop(100, 100, 24, 20).map(|v| (100.0 * (1.0 - v)).round());
    let psf = Image::from_rows(&[[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]).map(|v| v / 16.0);
    let x = richardson_lucy(&y, &psf, 20, 0.0);
    assert!(close(x.sum() / y.sum(), 1.0, 1e-3), "total of the estimate {} against the data {}: should be equal", x.sum(), y.sum());
    assert!(x.min() >= 0.0, "the estimate has a negative pixel, {}", x.min());
}

#[test]
fn c4_poisson_deviance() {
    let y = row([2., 0., 5., 5.]);
    let d = poisson_deviance(&y, &y);
    assert!(close(d, 0.0, 1e-6), "model = data: got {d}, expected 0");
    // pixel by pixel: 2(2 ln 2 − 1) = 0.7726, 2 · 1 = 2 (y = 0), 0, 0
    let d = poisson_deviance(&y, &row([1., 1., 5., 5.]));
    assert!(close(d, (0.7726 + 2.0) / 4.0, 1e-4), "got {d}, expected (0.7726 + 2) / 4 = 0.6932");
}
