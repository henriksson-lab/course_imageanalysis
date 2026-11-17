// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p shrinkage

use imgcourse::{samples, Image};
use shrinkage::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_hard_and_soft_b1() {
    let w = [5.0, -1.5, 0.4, -3.0, 2.1, -0.2];
    let hard_expected = [5.0, 0.0, 0.0, -3.0, 2.1, 0.0];
    let soft_expected = [3.0, 0.0, 0.0, -1.0, 0.1, 0.0];
    for i in 0..w.len() {
        assert!(close(hard(w[i], 2.0), hard_expected[i], 1e-5), "hard({}, 2): got {}, expected {} (B1)", w[i], hard(w[i], 2.0), hard_expected[i]);
        assert!(close(soft(w[i], 2.0), soft_expected[i], 1e-5), "soft({}, 2): got {}, expected {} (B1)", w[i], soft(w[i], 2.0), soft_expected[i]);
    }
}

#[test]
fn c2_noise_estimate_and_threshold_b2_b3() {
    let d = [0.8, -2.1, 0.3, 14.0, -1.2, 0.5, -0.9, 1.6, -25.0];
    let s = median_sigma(&d);
    assert!(close(s, 1.779, 2e-3), "σ̂ from the nine D coefficients: got {s}, expected 1.78 (B3a)");
    assert!(close(median_sigma(&[1.0, -3.0, 2.0, -4.0]), 2.5 / 0.6745, 1e-4), "an even count: the median of 1, 2, 3, 4 is 2.5");
    let t = universal_threshold(20.0, 65536);
    assert!(close(t, 94.2, 0.1), "universal threshold for σ = 20, N = 256²: got {t}, expected 94.2 (B2)");
    let t = universal_threshold(1.0, 1024);
    assert!(close(t, 3.723, 2e-3), "√(2 ln 1024): got {t}, expected 3.72 (B2)");
}

#[test]
fn c3_details_only() {
    let c = Image::from_fn(8, 8, |x, y| (x as f32 - 3.0) * (y as f32 + 1.0));
    let t = threshold_details(&c, 2, 5.0, true);
    for (x, y) in c.coords() {
        let expected = if x < 2 && y < 2 { c[(x, y)] } else { soft(c[(x, y)], 5.0) };
        assert!(close(t[(x, y)], expected, 1e-5), "position ({x}, {y}): got {}, expected {expected} (LL is the 2 × 2 corner for 2 levels of 8 × 8)", t[(x, y)]);
    }
}

#[test]
fn c4_edge_contrast_like_b4() {
    // As B4, in 2D: a vertical edge 0 | 100 on a 2 × 2 image, one level of Haar. V = −100.
    let img = Image::from_rows(&[[0.0, 100.0], [0.0, 100.0]]);
    let s = denoise(&img, &HAAR, 1, 20.0, true);
    assert!(close(s[(0, 0)], 10.0, 1e-3) && close(s[(1, 0)], 90.0, 1e-3), "soft, t = 20: got\n{s:.2}expected the edge 10 | 90 (contrast 100 − 2t)");
    let h = denoise(&img, &HAAR, 1, 20.0, false);
    assert!(close(h[(0, 0)], 0.0, 1e-3) && close(h[(1, 0)], 100.0, 1e-3), "hard, t = 20: got\n{h:.2}expected the edge unchanged");
    let h = denoise(&img, &HAAR, 1, 120.0, false);
    assert!(close(h[(0, 0)], 50.0, 1e-3) && close(h[(1, 0)], 50.0, 1e-3), "hard, t = 120: got\n{h:.2}expected 50 | 50 (the edge vanishes)");
    // t = 0 changes nothing.
    let cell = samples::cell().crop(0, 0, 64, 64);
    let same = denoise(&cell, &DB4, 3, 0.0, true);
    for (x, y) in cell.coords() {
        assert!(close(same[(x, y)], cell[(x, y)], 1e-4), "t = 0 must give the image back; pixel ({x}, {y}) is {}", same[(x, y)]);
    }
}

#[test]
fn c4_pure_noise_is_removed() {
    let flat = Image::new(64, 64, 0.5f32);
    let noisy = samples::add_noise(&flat, 0.1, 3);
    let t = universal_threshold(0.1, 64 * 64);
    let out = denoise(&noisy, &DB2, 3, t, false);
    let rms_in = (noisy.pixels().iter().map(|v| (v - 0.5).powi(2)).sum::<f32>() / 4096.0).sqrt();
    let rms_out = (out.pixels().iter().map(|v| (v - 0.5).powi(2)).sum::<f32>() / 4096.0).sqrt();
    assert!(rms_out < 0.2 * rms_in, "pure noise, universal threshold: RMS {rms_in} before, {rms_out} after; expected almost all noise removed");
}

#[test]
fn c5_cycle_spinning() {
    let img = samples::add_noise(&samples::cell().crop(50, 50, 32, 32), 0.05, 1);
    let one = cycle_spin(&img, &HAAR, 2, 0.1, true, 1);
    let plain = denoise(&img, &HAAR, 2, 0.1, true);
    for (x, y) in img.coords() {
        assert!(close(one[(x, y)], plain[(x, y)], 1e-5), "one shift must equal plain denoising; pixel ({x}, {y}): {} against {}", one[(x, y)], plain[(x, y)]);
    }
    // With all 4 × 4 shifts of a 2-level transform, shifting the input shifts the output.
    let shifted = Image::from_fn(32, 32, |x, y| img[((x + 1) % 32, y)]);
    let a = cycle_spin(&img, &HAAR, 2, 0.1, true, 4);
    let b = cycle_spin(&shifted, &HAAR, 2, 0.1, true, 4);
    for (x, y) in img.coords() {
        let expected = a[((x + 1) % 32, y)];
        assert!(close(b[(x, y)], expected, 1e-4), "16 shifts: denoising the shifted image should shift the result; pixel ({x}, {y}): {} against {expected}", b[(x, y)]);
    }
}
