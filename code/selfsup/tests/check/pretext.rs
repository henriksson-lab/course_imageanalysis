// Checks for the coding exercises of "Pretext tasks and masked image modelling".
// Run: cargo test -p selfsup pretext::

use imgcourse::Image;
use selfsup::pretext::*;

/// 2 × 2 windows that vary only along (1, 1, 1, 1): mean 1, t / 2 added to every pixel.
fn rank_one_windows() -> Vec<Vec<f64>> {
    [-1.0, 0.0, 1.0].iter().map(|t| vec![1.0 + t / 2.0; 4]).collect()
}

#[test]
fn c1_fill_interp_keeps_visible_and_fills_constant() {
    let img = Image::new(12, 12, 0.3f32);
    let mut mask = Image::new(12, 12, 0.0f32);
    for y in 4..8 {
        for x in 4..8 {
            mask[(x, y)] = 1.0;
        }
    }
    let noisy_inside = img.map_xy(|x, y, v| if mask[(x, y)] > 0.5 { 0.9 } else { v });
    let out = fill_interp(&noisy_inside, &mask, 2.0);
    for (x, y) in out.coords() {
        assert!((out[(x, y)] - 0.3).abs() < 1e-5, "a constant image with a hole is filled with the constant (got {} at ({x}, {y}))", out[(x, y)]);
    }
}

#[test]
fn c2_windows() {
    let img = Image::from_fn(10, 10, |x, y| (10 * y + x) as f32);
    let w = windows(&img, 4, 3);
    assert_eq!(w.len(), 9, "corners at 0, 3, 6 in each direction: 9 windows");
    assert_eq!(w[0][..5], [0.0, 1.0, 2.0, 3.0, 10.0], "the first window, row by row");
    assert_eq!(w[1][0], 3.0, "the second window starts at x = 3");
    assert_eq!(w[3][0], 30.0, "the fourth window starts at y = 3");
}

#[test]
fn c3_learn_model_rank_one() {
    let m = learn_model(&rank_one_windows(), 2, 1);
    assert!(m.mean.iter().all(|v| (v - 1.0).abs() < 1e-12), "mean window 1, got {:?}", m.mean);
    assert!(m.comps[0].iter().all(|v| (v.abs() - 0.5).abs() < 1e-6), "the component is ±(0.5, 0.5, 0.5, 0.5), got {:?}", m.comps[0]);
    // projections t on (0.5, 0.5, 0.5, 0.5): -1, 0, 1, variance 2/3
    assert!((m.variances[0] - 2.0 / 3.0).abs() < 1e-6, "variance along it 2/3, got {}", m.variances[0]);
}

#[test]
fn c4_complete_window() {
    let m = learn_model(&rank_one_windows(), 2, 1);
    // a new window 1.4 everywhere (t = 0.8); pixels 1 and 3 hidden (their values ignored)
    let full = complete_window(&m, &[1.4, 99.0, 1.4, -5.0], &[true, false, true, false], 1e-9);
    assert!(full.iter().all(|v| (v - 1.4).abs() < 1e-6), "the hidden pixels follow the visible ones: 1.4, got {full:?}");
    let m0 = WindowModel { comps: vec![], variances: vec![], ..m.clone() };
    let full = complete_window(&m0, &[1.4, 99.0, 1.4, -5.0], &[true, false, true, false], 1e-9);
    assert!(full.iter().all(|v| (v - 1.0).abs() < 1e-12), "k = 0: the mean window, got {full:?}");
    // a strong ridge pulls the coefficient towards 0: halfway when lambda / variance = number of visible pixels · 0.25
    let full = complete_window(&m, &[1.4, 99.0, 1.4, -5.0], &[true, false, true, false], 0.5 * 2.0 / 3.0);
    assert!((full[1] - 1.2).abs() < 1e-6, "B4: with lambda / variance = 0.5 the fit goes halfway: 1.2, got {}", full[1]);
}

#[test]
fn c5_hidden_psnr() {
    let truth = Image::new(4, 1, 0.5f32);
    let result = Image::from_rows(&[[0.6f32, 0.4, 0.9, 0.0]]);
    let mask = Image::from_rows(&[[1.0f32, 1.0, 0.0, 0.0]]);
    let p = hidden_psnr(&result, &truth, &mask);
    assert!((p - 20.0).abs() < 1e-3, "errors 0.1 on the two hidden pixels: 20 dB, got {p} (the visible pixels must not count)");
}
