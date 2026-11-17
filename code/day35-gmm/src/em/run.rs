// Runs your EM on the six points of B1, on a collapse, and on 1000 grey values drawn from a
// mixture of three classes, choosing K with BIC and AIC; pictures in out/em/.
//   cargo run -p day35 -- em                      (your code)
//   cargo run -p day35 --features solution -- em  (the model solution)

use crate::em::*;
use imgcourse::*;

/// A plot of the data's histogram (grey bars, values `lo..hi` in `bins` bins) with the
/// weighted components of the mixture (coloured) and their sum (black), scaled as counts.
fn plot(data: &[f32], m: &Mixture, lo: f32, hi: f32, bins: usize) -> Image<[f32; 3]> {
    let (w, h) = (4 * bins, 240);
    let width = (hi - lo) / bins as f32;
    let mut counts = vec![0.0f32; bins];
    for &x in data {
        if x >= lo && x < hi {
            counts[((x - lo) / width) as usize] += 1.0;
        }
    }
    // expected count in a bin = n · width · density
    let scale = data.len() as f32 * width;
    let top = 1.15 * counts.iter().cloned().fold(0.0, f32::max);
    let row = |v: f32| ((h - 1) as f32 * (1.0 - v / top)).clamp(0.0, (h - 1) as f32) as usize;
    let mut img = Image::from_fn(w, h, |px, py| if py >= row(counts[px / 4]) { [0.8; 3] } else { [1.0; 3] });
    let colour = |k: usize| io::label_colours(&Image::new(1, 1, k as u32 + 1))[(0, 0)];
    for px in 0..w {
        let x = lo + (px as f32 + 0.5) / 4.0 * width;
        let mut total = 0.0;
        for k in 0..m.weights.len() {
            let v = scale * m.weights[k] * normal(x, m.means[k], m.variances[k]);
            total += v;
            let y = row(v);
            img[(px, y)] = colour(k);
            img[(px, y.saturating_sub(1))] = colour(k);
        }
        let y = row(total);
        img[(px, y)] = [0.0; 3];
        if y > 0 {
            img[(px, y - 1)] = [0.0; 3];
        }
    }
    img
}

/// A random start: K different data points as means, the variance of all data for every
/// component, equal weights.
fn random_start(data: &[f32], k: usize, rng: &mut Rng) -> Mixture {
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let var = data.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / data.len() as f32;
    Mixture { weights: vec![1.0 / k as f32; k], means: (0..k).map(|_| data[rng.below(data.len())]).collect(), variances: vec![var; k] }
}

fn show(m: &Mixture) -> String {
    let parts: Vec<String> = (0..m.weights.len()).map(|k| format!("π {:.2} μ {:.2} σ² {:.3}", m.weights[k], m.means[k], m.variances[k])).collect();
    parts.join(" | ")
}

pub fn main() {
    let dir = "out/em";

    // 1. The six points of B1, from μ = (2, 8), σ² = (4, 4).
    let b1 = [1.0, 2.0, 4.0, 7.0, 9.0, 10.0];
    let start = Mixture { weights: vec![0.5, 0.5], means: vec![2.0, 8.0], variances: vec![4.0, 4.0] };
    let (m, trace) = em(&b1, &start, 1e-3, 1e-5, 1000);
    let shown: Vec<String> = trace.iter().take(6).map(|l| format!("{l:.2}")).collect();
    println!("B1: log p = {} ... {:.3} after {} E-steps", shown.join(", "), trace.last().unwrap(), trace.len());
    println!("  converged: {}", show(&m));

    // 2. Collapse: the same points, a third component on the point 10 with a small variance.
    let start = Mixture { weights: vec![0.4, 0.4, 0.2], means: vec![2.0, 8.0, 10.0], variances: vec![4.0, 4.0, 0.1] };
    for floor in [1e-6, 0.5] {
        let (m, trace) = em(&b1, &start, floor, 1e-5, 50);
        let shown: Vec<String> = trace.iter().take(6).map(|l| format!("{l:.2}")).collect();
        println!("collapse, variance floor {floor}: log p = {} ...", shown.join(", "));
        println!("  after {} E-steps: {}", trace.len(), show(&m));
    }

    // 3. 1000 grey values from the lecture's three classes: background (0.6, mean 60, σ 15),
    //    cytoplasm (0.3, 130, 20) and nucleus (0.1, 200, 15). Fit K = 1 to 5, best of 10 starts.
    let mut rng = Rng::new(8);
    let data: Vec<f32> = (0..1000)
        .map(|_| {
            let u = rng.uniform();
            let (mu, sd) = if u < 0.6 { (60.0, 15.0) } else if u < 0.9 { (130.0, 20.0) } else { (200.0, 15.0) };
            mu + sd * rng.normal()
        })
        .collect();
    println!("1000 grey values, best of 10 random starts:");
    for k in 1..=5 {
        let mut best: Option<(Mixture, f32)> = None;
        let mut lls = Vec::new();
        for _ in 0..10 {
            let (m, trace) = em(&data, &random_start(&data, k, &mut rng), 1.0, 1e-4, 2000);
            let ll = *trace.last().unwrap();
            lls.push(ll);
            if best.as_ref().is_none_or(|(_, b)| ll > *b) {
                best = Some((m, ll));
            }
        }
        let (m, ll) = best.unwrap();
        let (bic, aic) = bic_aic(ll, parameters(k), data.len());
        let worst = lls.iter().cloned().fold(f32::INFINITY, f32::min);
        println!("  K = {k}: log p = {ll:.1} (worst start {worst:.1}), p = {:>2}, BIC {bic:.1}, AIC {aic:.1}", parameters(k));
        if k == 3 {
            println!("    {}", show(&m));
        }
        save_rgb(&plot(&data, &m, 0.0, 256.0, 64), format!("{dir}/grey-k{k}.png"));
    }
    println!("wrote the pictures to {dir}/");
}
