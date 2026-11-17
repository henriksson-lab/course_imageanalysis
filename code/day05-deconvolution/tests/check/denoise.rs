// Checks for the coding exercises, using the numbers of the by-hand exercises.
// Run: cargo test -p day05 denoise::

use day05::denoise::*;
use imgcourse::{samples, Image};

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn check_row(what: &str, got: &Image<f32>, expected: &[f32]) {
    for (i, (g, e)) in got.row(0).iter().zip(expected).enumerate() {
        assert!(close(*g, *e, 1e-3), "{what}, pixel {}: got {g}, expected {e} (all: {:?})", i + 1, got.row(0));
    }
}

#[test]
fn c1_median_b1() {
    // A single row: the window of 3 × 3 sees the row itself three times (mirror border).
    let row = Image::from_rows(&[[10., 10., 10., 200., 10., 10., 50., 50., 50.]]);
    check_row("median of B1's row", &median_filter(&row, 1), &[10., 10., 10., 10., 10., 10., 50., 50., 50.]);
    let line = Image::from_rows(&[[10., 10., 10., 10., 50., 10., 10., 10., 10.]]);
    check_row("median of the 1-pixel line (B1c)", &median_filter(&line, 1), &[10.; 9]);
}

#[test]
fn c1_median_removes_salt() {
    let mut img = Image::new(7, 7, 0.3);
    img[(3, 3)] = 1.0;
    img[(1, 5)] = 0.0;
    let out = median_filter(&img, 1);
    for (x, y) in img.coords() {
        assert!(close(out[(x, y)], 0.3, 1e-6), "pixel ({x}, {y}): got {}, expected 0.3 (the outliers removed)", out[(x, y)]);
    }
}

#[test]
fn c2_bilateral_b2() {
    // B2: centre 10, neighbours 10 and 50, distance does not matter (huge σs), σr = 10.
    let row = Image::from_rows(&[[10., 10., 50.]]);
    let out = bilateral(&row, 1, 1e6, 10.0);
    assert!(close(out[(1, 0)], 10.007, 1e-3), "σr = 10: got {}, expected 10.007", out[(1, 0)]);
    // σr = 34: the 50 gets weight 0.5, so the centre becomes (10 + 10 + 0.5 · 50) / 2.5 = 18
    let sigma_r = (1600.0f32 / (2.0 * 2f32.ln())).sqrt();
    let out = bilateral(&row, 1, 1e6, sigma_r);
    assert!(close(out[(1, 0)], 18.0, 1e-2), "σr = {sigma_r}: got {}, expected 18", out[(1, 0)]);
}

#[test]
fn c2_bilateral_large_sigma_r_is_gaussian() {
    // With a huge σr every neighbour is "similar": a Gaussian-weighted mean.
    let img = samples::cell().crop(10, 10, 9, 9);
    let out = bilateral(&img, 2, 1.0, 1e6);
    let mut sum = 0.0;
    let mut total = 0.0;
    for dy in -2..=2isize {
        for dx in -2..=2isize {
            let w = (-((dx * dx + dy * dy) as f32) / 2.0).exp();
            sum += w * img[((4 + dx) as usize, (4 + dy) as usize)];
            total += w;
        }
    }
    assert!(close(out[(4, 4)], sum / total, 1e-5), "σr huge, pixel (4, 4): got {}, expected the Gaussian mean {}", out[(4, 4)], sum / total);
}

#[test]
fn c3_nlm_small_h_keeps_the_image() {
    let img = samples::cell().crop(50, 50, 12, 10);
    let out = nonlocal_means(&img, 1, 2, 1e-4);
    for (x, y) in img.coords() {
        assert!(close(out[(x, y)], img[(x, y)], 1e-4), "h tiny, pixel ({x}, {y}): got {}, expected {} (only the pixel itself matches)", out[(x, y)], img[(x, y)]);
    }
}

#[test]
fn c3_nlm_averages_repeated_texture() {
    // A lattice of dots, repeated every 4 pixels, with noise: similar patches are everywhere.
    let truth = Image::from_fn(24, 24, |x, y| if x % 4 == 1 && y % 4 == 1 { 0.8 } else { 0.2 });
    let noisy = samples::add_noise(&truth, 0.1, 3);
    let out = nonlocal_means(&noisy, 1, 5, 0.1);
    let err = |a: &Image<f32>| (a.pixels().iter().zip(truth.pixels()).map(|(p, q)| (p - q).powi(2)).sum::<f32>() / 576.0).sqrt();
    assert!(err(&out) < 0.75 * err(&noisy), "RMSE {} after, {} before: expected below 3/4 of it", err(&out), err(&noisy));
}
