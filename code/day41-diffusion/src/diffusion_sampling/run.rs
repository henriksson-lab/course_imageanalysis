// Deterministic sampling of 1D data (two modes at ±1.5) with the exact denoiser: 1 to 100
// steps, three step spacings, judged by spread, Fréchet distance, precision and recall; then
// a memoriser that copies 20 training points, caught by the nearest-training distance.
// Results in out/diffusion-sampling/.
//   cargo run -p day41 -- diffusion-sampling                      (your code)
//   cargo run -p day41 --features solution -- diffusion-sampling  (the model solution)

use crate::diffusion_sampling::*;
use imgcourse::*;

const MODES: [f32; 2] = [-1.5, 1.5];
const SPREAD: f32 = 0.3;
const SIGMA_MAX: f32 = 80.0;
const SIGMA_MIN: f32 = 0.002;

/// The exact best denoiser of the two modes at noise level σ.
fn denoise(y: f32, sigma: f32) -> f32 {
    let v = SPREAD * SPREAD + sigma * sigma;
    let w: Vec<f32> = MODES.iter().map(|m| (-(y - m).powi(2) / (2.0 * v)).exp()).collect();
    let total: f32 = w.iter().sum();
    MODES.iter().zip(&w).map(|(m, wk)| wk / total * (m + SPREAD * SPREAD / v * (y - m))).sum()
}

fn draw_data(n: usize, rng: &mut Rng) -> Vec<f32> {
    (0..n).map(|_| MODES[rng.below(2)] + SPREAD * rng.normal()).collect()
}

/// How the levels are spaced between SIGMA_MAX and SIGMA_MIN.
#[derive(Clone, Copy)]
enum Spacing {
    Uniform,   // evenly in σ
    Geometric, // evenly in log σ (= evenly in log-SNR)
    Rho7,      // evenly in σ^(1/7): dense at low noise, used by modern samplers
}

/// The noise levels of `n` steps: `n` levels from SIGMA_MAX to SIGMA_MIN, then a last step
/// to 0. One step: SIGMA_MAX → 0.
fn levels(n: usize, spacing: Spacing) -> Vec<f32> {
    let mut s = vec![SIGMA_MAX];
    for k in 1..n {
        let t = k as f32 / (n - 1) as f32;
        s.push(match spacing {
            Spacing::Uniform => SIGMA_MAX + (SIGMA_MIN - SIGMA_MAX) * t,
            Spacing::Geometric => SIGMA_MAX * (SIGMA_MIN / SIGMA_MAX).powf(t),
            Spacing::Rho7 => (SIGMA_MAX.powf(1.0 / 7.0) + t * (SIGMA_MIN.powf(1.0 / 7.0) - SIGMA_MAX.powf(1.0 / 7.0))).powi(7),
        });
    }
    s.push(0.0);
    s
}

/// A histogram of the samples (blue bars, as a density) with the density of the data (black),
/// from −3 to 3; the top of the picture is density 1.
fn histogram(samples: &[f32]) -> Image<[f32; 3]> {
    let (w, h, bins) = (300, 150, 60);
    let to_px = |density: f32| ((density * (h - 1) as f32) as usize).min(h - 1);
    let mut img = Image::new(w, h, [1.0f32; 3]);
    let mut count = vec![0usize; bins];
    for &s in samples {
        let b = ((s + 3.0) / 6.0 * bins as f32).floor();
        if (0.0..bins as f32).contains(&b) {
            count[b as usize] += 1;
        }
    }
    for (b, &c) in count.iter().enumerate() {
        let top = to_px(c as f32 / (samples.len() as f32 * 6.0 / bins as f32));
        for x in b * w / bins..(b + 1) * w / bins - 1 {
            for y in 0..top {
                img[(x, h - 1 - y)] = [0.4, 0.55, 0.9];
            }
        }
    }
    for x in 0..w {
        let v = -3.0 + 6.0 * (x as f32 + 0.5) / w as f32;
        let p: f32 = MODES.iter().map(|m| 0.5 * (-(v - m).powi(2) / (2.0 * SPREAD * SPREAD)).exp() / (SPREAD * (2.0 * std::f32::consts::PI).sqrt())).sum();
        img[(x, h - 1 - to_px(p))] = [0.0, 0.0, 0.0];
    }
    img
}

fn report(name: &str, samples: &[f32], real: &[f32]) {
    let (_, sd) = mean_sd(samples);
    let off = samples.iter().filter(|&&s| MODES.iter().all(|m| (s - m).abs() > 3.0 * SPREAD)).count() as f32 / samples.len() as f32;
    let (p, r) = precision_recall(real, samples, 3);
    println!("{name:28} sd {sd:5.2}   off the data {:5.1} %   Fréchet {:7.4}   precision {p:4.2}   recall {r:4.2}", 100.0 * off, frechet_1d(samples, real));
}

pub fn main() {
    let dir = "out/diffusion-sampling";
    let mut rng = Rng::new(1);
    let real = draw_data(2000, &mut rng); // held-out real data, for judging
    let train = draw_data(200, &mut rng); // the training data (for the memorisation check)
    println!("real data: mean {:.3}, sd {:.2}", mean_sd(&real).0, mean_sd(&real).1);
    let starts: Vec<f32> = (0..2000).map(|_| SIGMA_MAX * rng.normal()).collect();

    for steps in [1, 2, 5, 10, 20, 100] {
        let s: Vec<f32> = starts.iter().map(|&y| euler_sample(y, &levels(steps, Spacing::Rho7), denoise)).collect();
        report(&format!("{steps:3} steps"), &s, &real);
        save_rgb(&histogram(&s), format!("{dir}/steps{steps}.png"));
    }
    for (name, spacing) in [("uniform in σ", Spacing::Uniform), ("uniform in log σ", Spacing::Geometric)] {
        let s: Vec<f32> = starts.iter().map(|&y| euler_sample(y, &levels(5, spacing), denoise)).collect();
        report(&format!("  5 steps, {name}"), &s, &real);
        save_rgb(&histogram(&s), format!("{dir}/steps5-{}.png", if matches!(spacing, Spacing::Uniform) { "uniform" } else { "log" }));
    }

    // A memoriser: copies of 20 of the 200 training points, with a tiny jitter.
    let copies: Vec<f32> = (0..2000).map(|_| train[rng.below(20)] + 0.0001 * rng.normal()).collect();
    report("memoriser (20 copies)", &copies, &real);
    let diffusion: Vec<f32> = starts.iter().map(|&y| euler_sample(y, &levels(100, Spacing::Rho7), denoise)).collect();
    let median = |mut v: Vec<f32>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    println!(
        "median distance to the nearest training point: held-out real {:.4}, diffusion samples {:.4}, memoriser {:.4}",
        median(nearest_distances(&real, &train)),
        median(nearest_distances(&diffusion, &train)),
        median(nearest_distances(&copies, &train))
    );
    save_rgb(&histogram(&copies), format!("{dir}/memoriser.png"));
    println!("wrote the results to {dir}/");
}
