// Checks for the coding exercises.
// Run: cargo test -p day10 smoothers::

use imgcourse::{samples, Image};
use day10::smoothers::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_psnr() {
    let truth = Image::new(4, 4, 0.5);
    let p = psnr(&truth.map(|v| v + 0.1), &truth);
    assert!(close(p, 20.0, 1e-3), "every pixel off by 0.1: got {p} dB, expected 20");
    let p = psnr(&truth.map(|v| v - 0.01), &truth);
    assert!(close(p, 40.0, 1e-2), "every pixel off by 0.01: got {p} dB, expected 40");
}

#[test]
fn c2_median_removes_outliers_and_thin_lines() {
    let mut img = Image::new(7, 7, 0.2);
    img[(3, 3)] = 1.0; // a hot pixel
    let out = median_filter(&img, 1);
    assert!(close(out[(3, 3)], 0.2, 1e-6), "a single hot pixel must go: got {}", out[(3, 3)]);
    let line = Image::from_fn(7, 7, |x, _| if x == 3 { 1.0 } else { 0.0 });
    let out = median_filter(&line, 1);
    assert!(close(out[(3, 3)], 0.0, 1e-6), "a 1-pixel line is thinner than the 3 × 3 window and must go: got {}", out[(3, 3)]);
    let step = samples::step(8, 8, 0.0, 1.0);
    assert_eq!(median_filter(&step, 1), step, "a straight step must stay exactly as it is");
}

#[test]
fn c2_median_at_the_corner() {
    // At the corner the window holds 4 pixels: 1, 2, 3, 10. The median is (2 + 3) / 2.
    let img = Image::from_rows(&[[1.0, 2.0, 7.0], [3.0, 10.0, 7.0], [7.0, 7.0, 7.0]]);
    let m = median_filter(&img, 1)[(0, 0)];
    assert!(close(m, 2.5, 1e-6), "corner: got {m}, expected 2.5");
}

#[test]
fn c3_soft_threshold_b6() {
    for (c, expected) in [(5.0, 3.0), (-5.0, -3.0), (1.5, 0.0), (-2.0, 0.0)] {
        let got = soft_threshold(c, 2.0);
        assert!(close(got, expected, 1e-6), "soft threshold of {c} with t = 2: got {got}, expected {expected}");
    }
}

#[test]
fn c3_haar() {
    let img = samples::cell().crop(30, 30, 16, 10);
    let same = haar_shrink(&img, 0.0);
    let diff = same.zip_map(&img, |a, b| (a - b).abs()).max();
    assert!(diff < 1e-5, "t = 0 must give the image back (the transform is inverted exactly), largest difference {diff}");
    let flat = haar_shrink(&img, 100.0);
    let a = (img[(4, 6)] + img[(5, 6)] + img[(4, 7)] + img[(5, 7)]) / 4.0;
    assert!(close(flat[(5, 7)], a, 1e-5), "a huge t keeps only the mean of each 2 × 2 block: got {}, expected {a}", flat[(5, 7)]);
}

#[test]
fn c4_threshold_errors() {
    let truth = Image::from_rows(&[[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]);
    let result = Image::from_rows(&[[0.6, 0.9, 0.4], [0.1, 0.2, 0.8]]);
    assert_eq!(threshold_errors(&result, &truth, 0.5), 2, "two pixels are on the wrong side of 0.5");
}
