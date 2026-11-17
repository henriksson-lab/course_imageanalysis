// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p diffusion score_matching::

use imgcourse::{Image, Rng};
use diffusion::score_matching::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_noise_levels_b1() {
    for (ab, snr, sigma) in [(0.8, 4.0, 0.5), (0.5, 1.0, 1.0), (0.2, 0.25, 2.0)] {
        let (s, g) = noise_level(ab);
        assert!(close(s, snr, 1e-4) && close(g, sigma, 1e-4), "ᾱ = {ab}: got SNR {s}, σ {g}, expected {snr} and {sigma} (B1)");
    }
}

#[test]
fn c2_conversions_b5() {
    let (x0, score, v) = from_noise(1.0, -1.0, 0.64);
    assert!(close(x0, 2.0, 1e-4), "B5: clean estimate got {x0}, expected 2.0");
    assert!(close(score, 1.0 / 0.6, 1e-4), "B5: score got {score}, expected 1.67");
    assert!(close(v, -2.0, 1e-4), "B5: velocity got {v}, expected −2.0");
}

#[test]
fn c3_two_bins_b4() {
    let pairs = [(-2.1, -1.0), (-1.3, -1.0), (-0.6, -1.0), (-0.4, 1.0), (0.3, -1.0), (0.9, 1.0), (1.2, 1.0), (2.4, 1.0)];
    let bins = binned_average(&pairs, -3.0, 3.0, 2);
    assert!(close(bins[0].0, -1.5, 1e-5) && close(bins[1].0, 1.5, 1e-5), "bin centres: got {} and {}, expected −1.5 and 1.5", bins[0].0, bins[1].0);
    assert!(close(bins[0].1, -0.5, 1e-5) && close(bins[1].1, 0.5, 1e-5), "B4a: got {} and {}, expected −0.5 and +0.5", bins[0].1, bins[1].1);
    assert_eq!((bins[0].2, bins[1].2), (4, 4), "B4a: 4 pairs in each bin");
}

#[test]
fn c3_many_pairs_give_tanh_b4() {
    // B4c: with many pairs the bin averages approach the exact denoiser tanh(x)
    let mut rng = Rng::new(1);
    let pairs: Vec<(f32, f32)> = (0..40_000)
        .map(|_| {
            let x0 = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
            (x0 + rng.normal(), x0)
        })
        .collect();
    for (centre, mean, _) in binned_average(&pairs, -2.0, 2.0, 8) {
        // the bin is 0.5 wide: compare with the average of tanh over the bin
        let exact: f32 = (0..50).map(|k| (centre - 0.25 + (k as f32 + 0.5) * 0.01).tanh()).sum::<f32>() / 50.0;
        assert!(close(mean, exact, 0.04), "bin at {centre}: got {mean:.3}, expected about {exact:.3} (tanh, B4c)");
    }
}

#[test]
fn c4_power_of_white_noise_and_a_wave() {
    let mut rng = Rng::new(2);
    let noise = Image::from_fn(64, 64, |_, _| 2.0 * rng.normal());
    let p = radial_power(&noise);
    assert_eq!(p.len(), 46, "a 64 × 64 image has rings r = 0 ... 45 (the corners: 64 / √2 = 45.3)");
    let mean = p[5..].iter().sum::<f32>() / p[5..].len() as f32;
    assert!(close(mean, 4.0, 0.3), "white noise of sd 2: got mean power {mean}, expected 4 (its variance)");
    let wave = Image::from_fn(64, 64, |x, _| (2.0 * std::f32::consts::PI * 4.0 * x as f32 / 64.0).cos());
    let p = radial_power(&wave);
    let top = (0..p.len()).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
    assert_eq!(top, 4, "a cosine with 4 periods across: got the most power at r = {top}, expected r = 4");
}

#[test]
fn c5_wiener_gain_b3() {
    // a flat model of power 4 and noise σ = 1, 2, 4: the gain is 0.8, 0.5, 0.2 at every frequency (B3)
    let mut rng = Rng::new(3);
    let img = Image::from_fn(32, 32, |_, _| rng.normal());
    for (sigma, gain) in [(1.0, 0.8), (2.0, 0.5), (4.0, 0.2)] {
        let out = wiener_denoise(&img, sigma, &[4.0; 24]);
        for (x, y) in [(0, 0), (5, 7), (31, 20)] {
            assert!(close(out[(x, y)], gain * img[(x, y)], 1e-4), "σ = {sigma}, pixel ({x}, {y}): got {}, expected {gain} × {}", out[(x, y)], img[(x, y)]);
        }
    }
}
