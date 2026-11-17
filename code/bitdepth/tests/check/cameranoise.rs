// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p bitdepth cameranoise::

use bitdepth::cameranoise::*;
use imgcourse::{Image, Rng};

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

/// The camera of B1: full well 15 000 electrons, read noise 3 electrons, 12-bit ADC.
const B1: Camera = Camera { full_well: 15000.0, read_noise: 3.0, bits: 12 };

#[test]
fn c1_gain_and_dynamic_range_b1() {
    let g = gain(&B1);
    assert!(close(g, 3.663, 0.001), "B1(a): gain {g}, expected 15 000 / 4095 = 3.66 electrons per ADU");
    let dr = dynamic_range(&B1);
    assert!(close(dr, 5000.0, 0.01), "B1(b): dynamic range {dr}, expected 5000");
    let db = amplitude_db(dr);
    assert!(close(db, 74.0, 0.05), "B1(b): {db} dB, expected 74.0");
    let snr = amplitude_db(400.0 / 20.0);
    assert!(close(snr, 26.0, 0.05), "SNR 20 (400 electrons, shot noise 20): {snr} dB, expected 26.0");
}

#[test]
fn c2_poisson_mean_equals_variance() {
    let mut rng = Rng::new(1);
    for mean in [3.0f32, 25.0, 400.0] {
        let samples: Vec<f32> = (0..40000).map(|_| poisson(mean, &mut rng) as f32).collect();
        let (m, sd) = mean_and_sd(&samples);
        assert!(close(m, mean, 0.02 * mean), "Poisson with mean {mean}: sample mean {m}");
        assert!(close(sd * sd, mean, 0.04 * mean), "Poisson with mean {mean}: sample variance {}, expected {mean}", sd * sd);
    }
    let small: Vec<u32> = (0..1000).map(|_| poisson(3.0, &mut rng)).collect();
    assert!(small.contains(&0), "Poisson with mean 3: a count of 0 should happen (about 5 % of the time)");
}

#[test]
fn c3_expose_b7() {
    // B7: at 100 electrons the camera of B1 has noise √(100 + 9 + 1.06²) ≈ 10.5 electrons.
    let mut rng = Rng::new(2);
    let flat = Image::new(200, 200, 100.0f32);
    let adu = expose(&B1, &flat, &mut rng);
    let electrons: Vec<f32> = adu.pixels().iter().map(|&a| a as f32 * gain(&B1)).collect();
    let (m, sd) = mean_and_sd(&electrons);
    assert!(close(m, 100.0, 0.5), "B7: mean {m} electrons, expected 100");
    assert!(close(sd, 10.5, 0.3), "B7: measured noise {sd} electrons, expected about 10.5");
    let p = predicted_noise(&B1, 100.0);
    assert!(close(p, 10.5, 0.05), "B7: predicted noise {p}, expected 10.5");
    // Above the full well every pixel saturates: only the read noise (3 electrons, under 1 ADU) is left.
    let bright = expose(&B1, &Image::new(50, 50, 20000.0f32), &mut rng);
    assert!(bright.pixels().iter().all(|&a| a >= 4092), "a full well of 15 000 at 20 000 electrons: every pixel should read 4092 … 4095");
    // E2: an 8-bit ADC on a 30 000 electron well reads a dim flat patch as all the same number.
    let coarse = Camera { full_well: 30000.0, read_noise: 1.5, bits: 8 };
    let dim = expose(&coarse, &Image::new(50, 50, 5.0f32), &mut rng);
    assert!(dim.pixels().iter().all(|&a| a == 0), "8-bit ADC, 5 electrons: every pixel should read 0 ADU");
}

#[test]
fn c4_psnr_b4_b5() {
    // An error of ±5 everywhere has MSE 25.
    let reference = Image::new(10, 10, 100.0f32);
    let noisy = reference.map_xy(|x, y, v| if (x + y) % 2 == 0 { v + 5.0 } else { v - 5.0 });
    let p = psnr(&noisy, &reference, 255.0);
    assert!(close(p, 34.2, 0.05), "B4(a): σ = 5, peak 255: PSNR {p}, expected 34.2");
    let half = reference.map_xy(|x, y, v| if (x + y) % 2 == 0 { v + 2.5 } else { v - 2.5 });
    let p = psnr(&half, &reference, 255.0);
    assert!(close(p, 40.2, 0.05), "B4(b): σ = 2.5: PSNR {p}, expected 40.2");
    let n80 = reference.map_xy(|x, y, v| if (x + y) % 2 == 0 { v + 80.0 } else { v - 80.0 });
    let (a, b) = (psnr(&n80, &reference, 4095.0), psnr(&n80, &reference, 65535.0));
    assert!(close(a, 34.2, 0.05) && close(b, 58.3, 0.05), "B5: noise 80, peak 4095 gives {a} (expected 34.2), peak 65 535 gives {b} (expected 58.3)");
}
