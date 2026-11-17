// Checks for the coding exercises of "Self-supervised denoising".
// Run: cargo test -p selfsup self_denoise::

use imgcourse::{filters, samples, Image};
use selfsup::common::noise;
use selfsup::self_denoise::*;

fn grid() -> Image<f32> {
    Image::from_rows(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
}

fn noisy_crop(seed: u64) -> (Image<f32>, Image<f32>) {
    let clean = samples::cell().crop(64, 64, 64, 64);
    let n = noise(64, 64, 0.1, 0.0, seed);
    let noisy = clean.zip_map(&n, |a, b| a + b);
    (clean, noisy)
}

#[test]
fn c1_neighbourhood() {
    let v = neighbourhood(&grid(), 1, 1, 1, false);
    assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], "the 3 x 3 window around the centre, row by row");
    let v = neighbourhood(&grid(), 1, 1, 1, true);
    assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0], "blind: the centre value 5 is left out");
    let v = neighbourhood(&grid(), 0, 0, 1, false);
    assert_eq!(v, vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 4.0, 4.0, 5.0], "mirror border at the corner (0, 0): the edge pixel is repeated");
}

#[test]
fn c2_without_blind_spot_learns_the_identity() {
    let (_, noisy) = noisy_crop(3);
    let f = fit_filter(&noisy, &noisy, 1, false);
    assert_eq!(f.weights.len(), 9, "a 3 x 3 filter has 9 weights");
    assert!((f.weight(0, 0) - 1.0).abs() < 1e-3, "input = target, no blind spot: the best prediction is the pixel itself (centre weight 1), got {}", f.weight(0, 0));
    for (k, w) in f.weights.iter().enumerate() {
        if k != 4 {
            assert!(w.abs() < 1e-3, "input = target, no blind spot: every other weight should be 0, got {w} at index {k}");
        }
    }
}

#[test]
fn c2_blind_spot_filter_denoises() {
    let (clean, noisy) = noisy_crop(3);
    let f = fit_filter(&noisy, &noisy, 2, true);
    assert_eq!(f.weights.len(), 25, "a 5 x 5 filter has 25 weights");
    assert_eq!(f.weight(0, 0), 0.0, "the blind spot: centre weight 0");
    let sum: f32 = f.weights.iter().sum();
    assert!((sum - 1.0).abs() < 0.1, "the weights should add up to about 1 (an average of the neighbours), got {sum}");
    let out = apply_filter(&noisy, &f);
    let (before, after) = (filters::psnr(&noisy, &clean, 1.0), filters::psnr(&out, &clean, 1.0));
    assert!(after > before + 0.5, "Noise2Void should improve the PSNR: noisy {before:.2} dB, filtered {after:.2} dB");
}

#[test]
fn c2_noise2noise_matches_supervised() {
    let (clean, noisy) = noisy_crop(3);
    let (_, noisy2) = noisy_crop(4);
    let n2n = apply_filter(&noisy, &fit_filter(&noisy, &noisy2, 1, false));
    let sup = apply_filter(&noisy, &fit_filter(&noisy, &clean, 1, false));
    let (a, b) = (filters::psnr(&n2n, &clean, 1.0), filters::psnr(&sup, &clean, 1.0));
    assert!((a - b).abs() < 0.3, "Noise2Noise ({a:.2} dB) should be almost as good as the clean target ({b:.2} dB)");
}

#[test]
fn c3_apply_filter() {
    let f = Filter { radius: 1, weights: vec![0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0], bias: 0.5 };
    let out = apply_filter(&grid(), &f);
    // centre: (2 + 4 + 6 + 8) / 4 + 0.5
    assert!((out[(1, 1)] - 5.5).abs() < 1e-6, "the mean of the four neighbours plus the bias: expected 5.5, got {}", out[(1, 1)]);
    // corner (0, 0), mirror border: neighbours 1 (above), 1 (left), 2 (right), 4 (below)
    assert!((out[(0, 0)] - 2.5).abs() < 1e-6, "corner with mirror border: expected 2.5, got {}", out[(0, 0)]);
}

#[test]
fn c4_mix_back_b6() {
    // B6: sigma = 0.1, residuals 0.2 and 0: mean squared residual 0.02, so c = 1 − 0.01 / 0.02 = 0.5
    let noisy = Image::from_rows(&[[0.6, 0.3]]);
    let pred = Image::from_rows(&[[0.4, 0.3]]);
    let (out, c) = mix_back(&noisy, &pred, 0.1);
    assert!((c - 0.5).abs() < 1e-5, "B6: c = 1 − sigma² / mean residual² = 0.5, got {c}");
    assert!((out[(0, 0)] - 0.5).abs() < 1e-5 && (out[(1, 0)] - 0.3).abs() < 1e-5, "B6: halfway between prediction and pixel, got {:?}", out.pixels());
    let (_, c) = mix_back(&noisy, &pred, 0.2);
    assert_eq!(c, 0.0, "a residual smaller than the noise: c is clipped at 0");
}
