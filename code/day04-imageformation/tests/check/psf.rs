// Checks for the coding exercises, using the numbers of the lecture and of the by-hand exercises.
// Run: cargo test -p day04 psf::

use imgcourse::{samples, Border, Image};
use day04::psf::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn check_row(got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len(), "length: got {}, expected {}", got.len(), expected.len());
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        assert!(close(*g, *e, 1e-5), "pixel {i}: got {g}, expected {e} (whole row: got {got:?}, expected {expected:?})");
    }
}

#[test]
fn c1_three_points_lecture() {
    // Slide "The image of a point": brightness 1, 2, 1 at positions 1, 4, 6, PSF (1, 2, 1)/4.
    let mut points = vec![0.0; 8];
    points[1] = 1.0;
    points[4] = 2.0;
    points[6] = 1.0;
    let image = image_of_points(&points, &[0.25, 0.5, 0.25]);
    check_row(&image, &[0.25, 0.5, 0.25, 0.5, 1.0, 0.75, 0.5, 0.25]);
    let total: f32 = image.iter().sum();
    assert!(close(total, 4.0, 1e-5), "total brightness: got {total}, expected 4 (blur moves light, it does not remove it)");
}

#[test]
fn c1_b1() {
    let image = image_of_points(&[0., 0., 3., 0., 0., 0., 1., 1., 0.], &[0.25, 0.5, 0.25]);
    check_row(&image, &[0., 0.75, 1.5, 0.75, 0., 0.25, 0.75, 0.75, 0.25]);
}

#[test]
fn c1_light_falls_off_the_end() {
    // A point at the very edge loses the part of its PSF that falls outside the row.
    let image = image_of_points(&[2., 0., 0.], &[0.25, 0.5, 0.25]);
    check_row(&image, &[1.0, 0.5, 0.0]);
}

#[test]
fn c2_gaussian_psf() {
    for sigma in [0.8, 2.0, 3.3] {
        let g = gaussian_psf(sigma);
        let r = (3.0 * sigma).ceil() as usize;
        assert_eq!((g.width(), g.height()), (2 * r + 1, 2 * r + 1), "σ = {sigma}: size {} × {}, expected {} × {}", g.width(), g.height(), 2 * r + 1, 2 * r + 1);
        assert!(close(g.sum(), 1.0, 1e-4), "σ = {sigma}: the PSF sums to {}, expected 1", g.sum());
        assert!(g[(r, r)] == g.max(), "σ = {sigma}: the peak is not in the middle");
        assert!(close(g[(r + 1, r)], g[(r, r + 1)], 1e-7) && close(g[(r - 1, r)], g[(r + 1, r)], 1e-7), "σ = {sigma}: not round");
        // one σ from the centre the Gaussian has fallen to e^(-1/2) of its peak
        if sigma == 2.0 {
            let ratio = g[(r + 2, r)] / g[(r, r)];
            assert!(close(ratio, (-0.5f32).exp(), 1e-4), "σ = 2: two pixels out the PSF is {ratio} of the peak, expected e^(-1/2) = 0.607");
        }
    }
}

#[test]
fn c2_motion_psf() {
    let m = motion_psf(5);
    assert_eq!((m.width(), m.height()), (5, 1), "motion over 5 pixels: size {} × {}, expected 5 × 1", m.width(), m.height());
    assert!(close(m.sum(), 1.0, 1e-6) && close(m[(0, 0)], 0.2, 1e-6), "every pixel should be 1/5, got {:?}", m.pixels());
}

#[test]
fn c3_image_of_a_point_is_the_psf() {
    // An asymmetric PSF shows whether it is placed the right way round:
    // a point at (5, 5) must come out as the PSF itself, centred on (5, 5).
    let psf = Image::from_rows(&[[0.0, 0.1, 0.0], [0.0, 0.4, 0.3], [0.0, 0.2, 0.0]]);
    let mut point = Image::new(11, 11, 0.0);
    point[(5, 5)] = 1.0;
    let out = blur(&point, &psf, Border::Zero);
    for (i, j) in psf.coords() {
        let (x, y) = (5 + i - 1, 5 + j - 1);
        assert!(close(out[(x, y)], psf[(i, j)], 1e-6), "pixel ({x}, {y}): got {}, expected {} (the PSF, not mirrored)", out[(x, y)], psf[(i, j)]);
    }
    assert!(close(out.sum(), 1.0, 1e-6), "total light: got {}, expected 1", out.sum());
}

#[test]
fn c3_blur_of_a_row_is_c1() {
    let row = [0., 0., 3., 0., 0., 0., 1., 1., 0.];
    let img = Image::from_rows(&[row]);
    let psf = Image::from_rows(&[[0.25, 0.5, 0.25]]);
    let out = blur(&img, &psf, Border::Zero);
    check_row(out.row(0), &image_of_points(&row, &[0.25, 0.5, 0.25]));
}

#[test]
fn c3_keeps_the_mean_with_wrap() {
    let img = samples::cell().crop(60, 80, 24, 20);
    let out = blur(&img, &gaussian_psf(1.5), Border::Wrap);
    assert!(close(out.mean(), img.mean(), 1e-4), "mean brightness: got {}, expected {} (a PSF of sum 1 keeps it)", out.mean(), img.mean());
}

#[test]
fn c4_otf_of_121() {
    // The waves "constant, period 4, period 2" are multiplied by 1, 0.5 and 0.
    let h = [0.25, 0.5, 0.25];
    for (k, e) in [(0.0, 1.0), (0.25, 0.5), (0.5, 0.0)] {
        assert!(close(otf(&h, k), e, 1e-5), "OTF of (1, 2, 1)/4 at {k} cycles/pixel: got {}, expected {e}", otf(&h, k));
    }
}

#[test]
fn c4_motion_zeros_b6() {
    // Motion over L = 5 pixels: zeros at 1/L and 2/L, negative in between.
    let h = [0.2; 5];
    for k in [0.2, 0.4] {
        assert!(close(otf(&h, k), 0.0, 1e-5), "motion over 5 pixels at k = {k}: got {}, expected 0", otf(&h, k));
    }
    assert!(otf(&h, 0.3) < -0.2, "motion over 5 pixels at k = 0.3: got {}, expected about −0.25 (reversed contrast)", otf(&h, 0.3));
}

#[test]
fn c5_lens_mtf_b4() {
    // NA 1.4, 520 nm: cut-off 5.38 cycles/µm.
    for (k, e) in [(0.0, 1.0), (1.0, 0.765), (2.5, 0.431), (4.0, 0.150), (5.56, 0.0)] {
        let got = lens_mtf(k, 1.4, 0.52);
        assert!(close(got, e, 2e-3), "NA 1.4, 520 nm, {k} cycles/µm: got {got}, expected {e}");
    }
}
