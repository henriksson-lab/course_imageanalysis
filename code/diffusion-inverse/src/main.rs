// Measured and imagined, in 1D: signals of 64 samples with one or two spots, measured by
// an ideal low-pass (the first few cosines) with noise. The prior is a set of 20 000
// training signals, so the posterior is exact. Posterior mean, 32 samples, the unseen part,
// data consistency, and the coverage of the 90 % band with the right and a wrong prior.
// Results in out/diffusion-inverse/.
//   cargo run -p diffusion-inverse                      (your code)
//   cargo run -p diffusion-inverse --features solution  (the model solution)

use diffusion_inverse::*;
use imgcourse::*;

const N: usize = 64;
const SIGMA_N: f32 = 0.1;

/// A signal with spots of the given heights and width at the given positions.
fn spots(positions: &[f32], heights: &[f32], width: f32) -> Vec<f32> {
    (0..N).map(|i| positions.iter().zip(heights).map(|(p, a)| a * (-(i as f32 - p).powi(2) / (2.0 * width * width)).exp()).sum()).collect()
}

/// A random signal of the prior. Half of the time one spot, of height 1 to 2 and width
/// `width` to 2 `width`; otherwise two spots of height 1 and width `width`, 2 to 12 apart.
fn draw_signal(width: f32, rng: &mut Rng) -> Vec<f32> {
    let p = 12.0 + 40.0 * rng.uniform();
    if rng.uniform() < 0.5 {
        spots(&[p], &[1.0 + rng.uniform()], width * (1.0 + rng.uniform()))
    } else {
        spots(&[p, p + 2.0 + 10.0 * rng.uniform()], &[1.0, 1.0], width)
    }
}

/// The number of spots: local maxima above 0.5.
fn count_spots(x: &[f32]) -> usize {
    (1..N - 1).filter(|&i| x[i] > 0.5 && x[i] >= x[i - 1] && x[i] > x[i + 1]).count()
}

/// The data residual in units of the noise: |y − A x| / (σ_n √(number of measurements)).
fn residual(y: &[f32], x: &[f32], keep: usize) -> f32 {
    let m = measure(x, keep);
    (y.iter().zip(&m).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / keep as f32).sqrt() / SIGMA_N
}

/// Signals drawn as lines, values −0.5 ... 1.5; the first in black (thick), the others in colour.
fn plot(lines: &[(&[f32], [f32; 3])]) -> Image<[f32; 3]> {
    let (w, h) = (320, 160);
    let mut img = Image::new(w, h, [1.0f32; 3]);
    let zero = h - 1 - ((0.5 / 2.0) * (h - 1) as f32) as usize;
    for x in 0..w {
        img[(x, zero)] = [0.85, 0.85, 0.85];
    }
    for &(line, colour) in lines {
        for px in 0..w {
            let t = px as f32 / (w - 1) as f32 * (N - 1) as f32;
            let i = (t as usize).min(N - 2);
            let v = line[i] + (t - i as f32) * (line[i + 1] - line[i]);
            let py = ((v + 0.5) / 2.0 * (h - 1) as f32).round();
            if (0.0..h as f32).contains(&py) {
                img[(px, h - 1 - py as usize)] = colour;
            }
        }
    }
    img
}

struct Prior {
    signals: Vec<Vec<f32>>,
    prior: Vec<f32>,
}

impl Prior {
    fn new(width: f32, seed: u64) -> Prior {
        let mut rng = Rng::new(seed);
        let signals: Vec<Vec<f32>> = (0..20_000).map(|_| draw_signal(width, &mut rng)).collect();
        let prior = vec![1.0 / signals.len() as f32; signals.len()];
        Prior { signals, prior }
    }

    /// 32 posterior samples (and the weights) for the measurement y.
    fn samples(&self, y: &[f32], keep: usize, rng: &mut Rng) -> (Vec<f32>, Vec<Vec<f32>>) {
        let measured: Vec<Vec<f32>> = self.signals.iter().map(|s| measure(s, keep)).collect();
        let w = posterior_weights(y, &measured, &self.prior, SIGMA_N);
        let s = (0..32).map(|_| self.signals[posterior_sample(&w, rng)].clone()).collect();
        (w, s)
    }
}

fn main() {
    let dir = "out/diffusion-inverse";
    let t0 = std::time::Instant::now();
    let matched = Prior::new(1.2, 1);
    let mut rng = Rng::new(7);
    let colours = [[0.15, 0.35, 0.85], [0.85, 0.3, 0.1], [0.1, 0.6, 0.2], [0.6, 0.2, 0.7], [0.9, 0.6, 0.0]];

    // Two spots 4 apart, measured with 24 and with 6 cosines.
    let truth = spots(&[30.0, 34.0], &[1.0, 1.0], 1.2);
    for keep in [24, 6] {
        let y: Vec<f32> = measure(&truth, keep).iter().map(|v| v + SIGMA_N * rng.normal()).collect();
        let (w, samples) = matched.samples(&y, keep, &mut rng);
        let mean = posterior_mean(&w, &matched.signals);
        let counts: Vec<usize> = samples.iter().map(|s| count_spots(s)).collect();
        let n2 = counts.iter().filter(|&&c| c == 2).count();
        let n1 = counts.iter().filter(|&&c| c == 1).count();
        let mut res: Vec<f32> = samples.iter().map(|s| residual(&y, s, keep)).collect();
        res.sort_by(|a, b| a.total_cmp(b));
        let effective = 1.0 / w.iter().map(|v| v * v).sum::<f32>();
        println!(
            "{keep:2} cosines kept: truth 2 spots; mean {} spot(s); samples: {n1} with one spot, {n2} with two; residual / expected: mean {:.2}, samples (median) {:.2}; about {effective:.0} training signals share the weight",
            count_spots(&mean),
            residual(&y, &mean, keep),
            res[16]
        );
        let mut lines: Vec<(&[f32], [f32; 3])> = vec![(&truth, [0.0, 0.0, 0.0])];
        for (k, s) in samples.iter().take(5).enumerate() {
            lines.push((s, colours[k]));
        }
        save_rgb(&plot(&lines), format!("{dir}/samples-keep{keep}.png"));
        save_rgb(&plot(&[(&truth, [0.0, 0.0, 0.0]), (&mean, colours[0])]), format!("{dir}/mean-keep{keep}.png"));
        let unseen: Vec<Vec<f32>> = samples.iter().take(5).map(|s| split_measured(s, keep).1).collect();
        let unseen_lines: Vec<(&[f32], [f32; 3])> = unseen.iter().enumerate().map(|(k, u)| (u.as_slice(), colours[k])).collect();
        save_rgb(&plot(&unseen_lines), format!("{dir}/unseen-keep{keep}.png"));
        let sd_unseen: f32 = (0..N)
            .map(|i| {
                let v: Vec<f32> = samples.iter().map(|s| split_measured(s, keep).1[i]).collect();
                let m = v.iter().sum::<f32>() / v.len() as f32;
                (v.iter().map(|a| (a - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
            })
            .fold(0.0, f32::max);
        let sd_seen: f32 = (0..N)
            .map(|i| {
                let v: Vec<f32> = samples.iter().map(|s| split_measured(s, keep).0[i]).collect();
                let m = v.iter().sum::<f32>() / v.len() as f32;
                (v.iter().map(|a| (a - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
            })
            .fold(0.0, f32::max);
        println!("    largest disagreement between the samples (sd at one position): seen part {sd_seen:.3}, unseen part {sd_unseen:.3}");
    }

    // The same measurement (6 cosines) with a wrong prior: spots twice as wide.
    let wider = Prior::new(2.4, 2);
    let y: Vec<f32> = measure(&truth, 6).iter().map(|v| v + SIGMA_N * rng.normal()).collect();
    let (_, samples) = wider.samples(&y, 6, &mut rng);
    let n1 = samples.iter().filter(|s| count_spots(s) == 1).count();
    let mut res: Vec<f32> = samples.iter().map(|s| residual(&y, s, 6)).collect();
    res.sort_by(|a, b| a.total_cmp(b));
    println!(" 6 cosines, prior with spots twice as wide: samples: {n1} with one spot, {} with two; residual / expected of the samples (median) {:.2}", samples.len() - n1, res[16]);
    let mut lines: Vec<(&[f32], [f32; 3])> = vec![(&truth, [0.0, 0.0, 0.0])];
    for (k, s) in samples.iter().take(5).enumerate() {
        lines.push((s, colours[k]));
    }
    save_rgb(&plot(&lines), format!("{dir}/samples-keep6-wider.png"));

    // Coverage on 100 truths from the matched prior (not in the training set), 6 cosines.
    let mut truths_rng = Rng::new(11);
    let truths: Vec<Vec<f32>> = (0..100).map(|_| draw_signal(1.2, &mut truths_rng)).collect();
    for (name, prior) in [("matched prior", &matched), ("prior with spots twice as wide", &wider)] {
        let keep = 6;
        let measured: Vec<Vec<f32>> = prior.signals.iter().map(|s| measure(s, keep)).collect();
        let mut cover = 0.0;
        let mut res = 0.0;
        for t in &truths {
            let y: Vec<f32> = measure(t, keep).iter().map(|v| v + SIGMA_N * rng.normal()).collect();
            let w = posterior_weights(&y, &measured, &prior.prior, SIGMA_N);
            let samples: Vec<Vec<f32>> = (0..32).map(|_| prior.signals[posterior_sample(&w, &mut rng)].clone()).collect();
            cover += band_coverage(t, &samples);
            res += residual(&y, &samples[0], keep);
        }
        println!("{name:30}: truth inside the 90 % band at {:.1} % of the positions; mean residual / expected of a sample {:.2}", cover, res / 100.0);
    }
    println!("wrote the results to {dir}/ in {:.1} s", t0.elapsed().as_secs_f32());
}
