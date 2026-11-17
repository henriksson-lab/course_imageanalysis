// Denoising score matching in 1D (training pairs, their binned average and the exact best
// denoiser), and the Wiener denoiser of the micrograph at three noise levels, with the
// crossover frequency where the image's power drops below the noise. Results in
// out/score-matching/.
//   cargo run -p diffusion -- score-matching                      (your code)
//   cargo run -p diffusion --features solution -- score-matching  (the model solution)

use imgcourse::*;
use crate::score_matching::*;

/// The data of the 1D part: two narrow modes at ±1.5 (sd 0.15), equally likely.
const MODES: [f32; 2] = [-1.5, 1.5];
const SPREAD: f32 = 0.15;

/// The exact best denoiser `E[x0 | x]` for the two modes at noise level σ.
fn exact_denoiser(x: f32, sigma: f32) -> f32 {
    let v = SPREAD * SPREAD + sigma * sigma;
    let w: Vec<f32> = MODES.iter().map(|m| (-(x - m).powi(2) / (2.0 * v)).exp()).collect();
    let total: f32 = w.iter().sum();
    MODES.iter().zip(&w).map(|(m, wk)| wk / total * (m + SPREAD * SPREAD / v * (x - m))).sum()
}

/// Training pairs (noisy, clean) at noise level σ.
fn pairs(n: usize, sigma: f32, rng: &mut Rng) -> Vec<(f32, f32)> {
    (0..n)
        .map(|_| {
            let x0 = MODES[rng.below(2)] + SPREAD * rng.normal();
            (x0 + sigma * rng.normal(), x0)
        })
        .collect()
}

/// A plot from −3 to 3 in both directions: the pairs (grey), the binned average (green) and
/// the exact denoiser (blue).
fn plot(train: &[(f32, f32)], bins: &[(f32, f32, usize)], sigma: f32) -> Image<[f32; 3]> {
    let size = 300;
    let mut img = Image::new(size, size, [1.0f32; 3]);
    let to_px = |v: f32| ((v + 3.0) / 6.0 * size as f32) as isize;
    let mut dot = |x: f32, y: f32, colour: [f32; 3], r: isize| {
        let (cx, cy) = (to_px(x), size as isize - 1 - to_px(y));
        for dy in -r..=r {
            for dx in -r..=r {
                if img.contains(cx + dx, cy + dy) {
                    img[((cx + dx) as usize, (cy + dy) as usize)] = colour;
                }
            }
        }
    };
    for &(x, x0) in train {
        dot(x, x0, [0.75, 0.75, 0.75], 1);
    }
    for k in 0..600 {
        let x = -3.0 + k as f32 * 0.01;
        dot(x, exact_denoiser(x, sigma), [0.15, 0.35, 0.85], 0);
    }
    let width = 6.0 / bins.len() as f32;
    for &(centre, mean, count) in bins {
        if count > 0 {
            for k in 0..30 {
                dot(centre - width / 2.0 + k as f32 * width / 30.0, mean, [0.1, 0.65, 0.2], 0);
            }
        }
    }
    img
}

pub fn main() {
    let dir = "out/score-matching";

    // 1. Pairs, binned average and the exact denoiser.
    let mut rng = Rng::new(1);
    for sigma in [0.15, 0.6, 2.5] {
        let train = pairs(400, sigma, &mut rng);
        let bins = binned_average(&train, -3.0, 3.0, 20);
        let sparse = bins.iter().filter(|b| b.2 < 3).count();
        // the mean squared denoising error on fresh pairs, of the exact denoiser and of the
        // binned average (bins without pairs: the noisy value itself)
        let test = pairs(4000, sigma, &mut rng);
        let lookup = |x: f32| {
            let k = (((x + 3.0) / 0.3).floor().max(0.0) as usize).min(19);
            if bins[k].2 > 0 { bins[k].1 } else { x }
        };
        let err_exact = test.iter().map(|&(x, x0)| (exact_denoiser(x, sigma) - x0).powi(2)).sum::<f32>() / test.len() as f32;
        let err_binned = test.iter().map(|&(x, x0)| (lookup(x) - x0).powi(2)).sum::<f32>() / test.len() as f32;
        println!("σ = {sigma:4}: {sparse:2} of 20 bins with fewer than 3 pairs; denoising error: exact {err_exact:.3}, binned average {err_binned:.3}; a noisy 1.5 goes to {:.2}", exact_denoiser(1.5, sigma));
        save_rgb(&plot(&train, &bins, sigma), format!("{dir}/pairs-sigma{sigma}.png"));
    }

    // 2. The micrograph, scaled to mean 0 and variance 1, at three noise levels.
    let crop = samples::cell().crop(64, 64, 128, 128);
    let mean = crop.mean();
    let sd = (crop.pixels().iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (128.0 * 128.0)).sqrt();
    let clean = crop.map(|v| (v - mean) / sd);
    let power = radial_power(&clean);
    let show = |img: &Image<f32>| img.map(|v| 0.5 + v / 6.0);
    save_gray(&show(&clean), format!("{dir}/clean.png"));
    for log_snr in [2.3f32, 0.0, -2.3, -4.6] {
        let alpha_bar = 1.0 / (1.0 + (-log_snr).exp()); // SNR = ᾱ / (1 − ᾱ) = e^log_snr
        let (snr, sigma) = noise_level(alpha_bar);
        let crossover = (1..power.len()).find(|&r| power[r] < sigma * sigma).unwrap_or(power.len()); // ring 0 is the mean, 0 here
        let noisy = samples::add_noise(&clean, sigma, 7);
        let denoised = wiener_denoise(&noisy, sigma, &power);
        let mse = |a: &Image<f32>| a.zip_map(&clean, |p, q| (p - q).powi(2)).mean();
        println!(
            "log-SNR {log_snr:4.1} (SNR {snr:6.2}, σ {sigma:5.2}): image above the noise up to r = {crossover:2} cycles per image; error noisy {:.3}, Wiener {:.3}",
            mse(&noisy),
            mse(&denoised)
        );
        save_gray(&show(&noisy), format!("{dir}/noisy-logsnr{log_snr}.png"));
        save_gray(&show(&denoised), format!("{dir}/wiener-logsnr{log_snr}.png"));
    }
    println!("wrote the results to {dir}/");
}
