// The forward process on the micrograph, sampling 2D point clouds with the exact
// denoiser (few and many steps, deterministic and stochastic, 20 training points), and the
// error of a posterior mean against a posterior sample. Results in out/diffusion/.
//   cargo run -p diffusion                      (your code)
//   cargo run -p diffusion --features solution  (the model solution)

use diffusion::*;
use imgcourse::*;

/// The signal levels for `n` steps from pure noise (τ = 1) to the data (τ = 0).
fn levels(n: usize) -> Vec<f32> {
    (0..=n).map(|k| cosine_alpha_bar(1.0 - k as f32 / n as f32)).collect()
}

/// A scatter plot from −2.5 to 2.5 in both directions: grey training data, blue samples.
fn plot(data: &[Vec<f32>], samples: &[Vec<f32>]) -> Image<[f32; 3]> {
    let size = 300;
    let mut img = Image::new(size, size, [1.0f32; 3]);
    let mut dot = |p: &[f32], colour: [f32; 3]| {
        let (cx, cy) = (((p[0] + 2.5) / 5.0 * size as f32) as isize, ((2.5 - p[1]) / 5.0 * size as f32) as isize);
        for d in 0..9 {
            let (x, y) = (cx + d % 3 - 1, cy + d / 3 - 1);
            if img.contains(x, y) {
                img[(x as usize, y as usize)] = colour;
            }
        }
    };
    for p in data {
        dot(p, [0.75, 0.75, 0.75]);
    }
    for p in samples {
        dot(p, [0.15, 0.35, 0.85]);
    }
    img
}

/// The fraction of samples further than `3 spread` from every blob centre: "off the data".
fn off_data(samples: &[Vec<f32>], centres: &[Vec<f32>], spread: f32) -> f32 {
    let off = samples
        .iter()
        .filter(|s| centres.iter().all(|c| ((s[0] - c[0]).powi(2) + (s[1] - c[1]).powi(2)).sqrt() > 3.0 * spread))
        .count();
    off as f32 / samples.len() as f32
}

fn main() {
    let dir = "out/diffusion";

    // 1. The forward process on the micrograph (grey values scaled to −1 ... 1).
    let cell = samples::cell().map(|v| 2.0 * v - 1.0);
    let mut rng = Rng::new(1);
    let noise: Vec<f32> = (0..256 * 256).map(|_| rng.normal()).collect();
    for tau in [0.0, 0.2, 0.4, 0.6, 1.0] {
        let ab = cosine_alpha_bar(tau);
        let noisy = Image::from_vec(256, 256, forward(cell.pixels(), ab, &noise));
        println!("τ = {tau}: signal {:.2}, noise {:.2}", ab.sqrt(), (1.0 - ab).sqrt());
        save_gray(&noisy.map(|v| 0.5 + 0.5 * v), format!("{dir}/forward-tau{tau}.png"));
    }

    // 2. Sampling two blobs and a ring of 8 blobs with the exact denoiser.
    let spread = 0.12;
    let two: Vec<Vec<f32>> = vec![vec![-1.2, 0.0], vec![1.2, 0.0]];
    let ring: Vec<Vec<f32>> = (0..8).map(|k| k as f32 * std::f32::consts::FRAC_PI_4).map(|t| vec![1.6 * t.cos(), 1.6 * t.sin()]).collect();
    let mut rng = Rng::new(2);
    let starts: Vec<Vec<f32>> = (0..500).map(|_| vec![rng.normal(), rng.normal()]).collect();
    for (name, centres) in [("two", &two), ("ring", &ring)] {
        // training data drawn from the blobs, for the picture
        let data: Vec<Vec<f32>> = (0..500).map(|k| {
            let c = &centres[k % centres.len()];
            vec![c[0] + spread * rng.normal(), c[1] + spread * rng.normal()]
        }).collect();
        for steps in [1, 2, 5, 100] {
            let s: Vec<Vec<f32>> = starts.iter().map(|x| sample_ddim(x, &levels(steps), centres, spread)).collect();
            println!("{name} blobs, deterministic, {steps:3} steps: {:5.1} % off the data", 100.0 * off_data(&s, centres, spread));
            save_rgb(&plot(&data, &s), format!("{dir}/{name}-ddim{steps}.png"));
        }
        let mut rng_s = Rng::new(3);
        let s: Vec<Vec<f32>> = starts.iter().map(|x| sample_ddpm(x, &levels(100), centres, spread, &mut rng_s)).collect();
        println!("{name} blobs, stochastic,    100 steps: {:5.1} % off the data", 100.0 * off_data(&s, centres, spread));
        save_rgb(&plot(&data, &s), format!("{dir}/{name}-ddpm100.png"));
    }

    // 3. The perfect denoiser of 20 training points (spread 0) only reproduces them.
    let train: Vec<Vec<f32>> = (0..20).map(|_| vec![1.2 * rng.normal(), 1.2 * rng.normal()]).collect();
    let s: Vec<Vec<f32>> = starts.iter().map(|x| sample_ddim(x, &levels(100), &train, 0.0)).collect();
    let copies = s.iter().filter(|p| train.iter().any(|t| ((p[0] - t[0]).powi(2) + (p[1] - t[1]).powi(2)).sqrt() < 0.05)).count();
    println!("20 training points, perfect denoiser: {copies} of 500 samples within 0.05 of a training point");
    save_rgb(&plot(&train, &s), format!("{dir}/memorise.png"));

    // 4. Restoration: posterior mean against a posterior sample (prior N(0, 1), noise sd 0.5).
    let mut rng = Rng::new(4);
    let (mut err_mean, mut err_sample, mut err_raw) = (0.0, 0.0, 0.0);
    let n = 100_000;
    for _ in 0..n {
        let x = rng.normal();
        let y = x + 0.5 * rng.normal();
        let (m, v) = posterior(0.0, 1.0, 0.5, y);
        let sample = m + v.sqrt() * rng.normal();
        err_raw += (y - x).powi(2) / n as f32;
        err_mean += (m - x).powi(2) / n as f32;
        err_sample += (sample - x).powi(2) / n as f32;
    }
    println!("restoring {n} values: mean squared error of the measurement {err_raw:.3}, posterior mean {err_mean:.3}, posterior sample {err_sample:.3}");
    println!("wrote the results to {dir}/");
}
