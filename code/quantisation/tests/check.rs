// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p quantisation

use imgcourse::Image;
use quantisation::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

/// A smooth ramp from black to white, 256 × 256, every pixel a different value.
fn ramp() -> Image<f32> {
    Image::from_fn(256, 256, |x, y| (y * 256 + x) as f32 / 65535.0)
}

#[test]
fn c1_quantise_b2() {
    let values = [0.10, 0.22, 0.37, 0.55, 0.81];
    let codes = [0, 1, 1, 2, 2];
    let errors = [-0.100, 0.113, -0.037, 0.117, -0.143];
    for i in 0..5 {
        let k = quantise(values[i], 2);
        assert_eq!(k, codes[i], "B2: {} to 2 bits: got code {k}, expected {}", values[i], codes[i]);
        let e = level(k, 2) - values[i];
        assert!(close(e, errors[i], 0.001), "B2: {} to 2 bits: error {e:.3}, expected {}", values[i], errors[i]);
    }
    assert_eq!(quantise(1.0, 10), 1023, "B1: 10 bits have 1024 levels, the top code is 1023");
    assert_eq!(quantise(1.3, 8), 255, "values above 1 should give the top code");
    assert_eq!(quantise(-0.2, 8), 0, "values below 0 should give code 0");
    let q = quantise_image(&Image::from_rows(&[[0.10, 0.81]]), 2);
    assert!(close(q[(0, 0)], 0.0, 1e-6) && close(q[(1, 0)], 2.0 / 3.0, 1e-6), "quantise_image: got {:?}, expected [0, 2/3]", q.pixels());
}

#[test]
fn c2_rms_and_snr_b4() {
    let img = ramp();
    for (bits, rms, snr) in [(6, 0.0046, 36.0), (10, 0.00028, 60.2)] {
        let q = quantise_image(&img, bits);
        let e = rms_error(&img, &q);
        let s = snr_db(&img, &q);
        assert!(close(e, rms, 0.03 * rms), "B4, ramp at {bits} bits: RMS error {e:.6}, expected about {rms}");
        assert!(close(s, snr, 0.3), "B4, ramp at {bits} bits: SNR {s:.1} dB, expected about {snr}");
    }
}

#[test]
fn c3_dither_b3() {
    // B3: a flat region of 0.25 at 2 bits.
    let flat = Image::new(128, 128, 0.25f32);
    let plain = quantise_image(&flat, 2);
    assert!(plain.pixels().iter().all(|&v| close(v, 1.0 / 3.0, 1e-6)), "B3(a): without dither every pixel should become 1/3");
    let d = dither_random(&flat, 2, 1);
    let up = d.pixels().iter().filter(|&&v| close(v, 1.0 / 3.0, 1e-6)).count() as f32 / d.pixels().len() as f32;
    let down = d.pixels().iter().filter(|&&v| v == 0.0).count() as f32 / d.pixels().len() as f32;
    assert!(close(up, 0.75, 0.02) && close(up + down, 1.0, 1e-6), "B3(b): {up:.3} of the pixels round up to 1/3 and {down:.3} down to 0, expected 0.75 and 0.25");
    assert!(close(d.mean(), 0.25, 0.005), "B3(b): the mean is {:.4}, expected 0.25", d.mean());
    // B6: on the ramp the dithered error is √2 times the plain one.
    let img = ramp();
    let ratio = rms_error(&img, &dither_random(&img, 4, 2)) / rms_error(&img, &quantise_image(&img, 4));
    assert!(close(ratio, 2f32.sqrt(), 0.05), "B6: dithered RMS / plain RMS = {ratio:.3}, expected about 1.41");
}

#[test]
fn c4_error_diffusion_keeps_local_means() {
    let flat = Image::new(64, 64, 0.25f32);
    let d = error_diffusion(&flat, 2);
    assert!(d.pixels().iter().all(|&v| [0.0, 1.0 / 3.0].iter().any(|&l| close(v, l, 1e-6))), "a flat 0.25 should only use the levels 0 and 1/3");
    // Every 8 × 8 block has (almost) the right mean.
    for by in 0..8 {
        for bx in 0..8 {
            let m = d.crop(8 * bx, 8 * by, 8, 8).mean();
            assert!(close(m, 0.25, 0.03), "block ({bx}, {by}): mean {m:.3}, expected about 0.25");
        }
    }
    let img = ramp();
    let e = error_diffusion(&img, 1);
    assert!(close(e.mean(), img.mean(), 0.002), "ramp at 1 bit: mean {:.4}, expected {:.4}", e.mean(), img.mean());
}
