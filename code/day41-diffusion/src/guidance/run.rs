// Guided sampling in 1D with exact denoisers: class A = N(−1, 1), class B = N(+1, 1),
// equally likely. For guidance weights w = 0, 1, 3, 8, deterministic and stochastic: the
// samples' mean and sd, the fraction beyond the most extreme 1 % of real class members,
// against the sharpened density. Then a class A with a sub-mode near B. Results in
// out/guidance/.
//   cargo run -p day41 -- guidance                      (your code)
//   cargo run -p day41 --features solution -- guidance  (the model solution)

use crate::guidance::*;
use imgcourse::*;

const STEPS: usize = 100;

/// A histogram of the samples (blue) from −6 to 4 with the class density (black) and the
/// sharpened density (red); the top of the picture is density 1.
fn histogram(samples: &[f32], w: f32) -> Image<[f32; 3]> {
    let (width, h, bins) = (400, 150, 80);
    let (lo, hi) = (-6.0, 4.0);
    let to_px = |d: f32| ((d * (h - 1) as f32) as usize).min(h - 1);
    let mut img = Image::new(width, h, [1.0f32; 3]);
    let bw = (hi - lo) / bins as f32;
    let mut count = vec![0usize; bins];
    for &s in samples {
        let b = ((s - lo) / bw).floor();
        if (0.0..bins as f32).contains(&b) {
            count[b as usize] += 1;
        }
    }
    for (b, &c) in count.iter().enumerate() {
        let top = to_px(c as f32 / (samples.len() as f32 * bw));
        for x in b * width / bins..(b + 1) * width / bins - 1 {
            for y in 0..top {
                img[(x, h - 1 - y)] = [0.4, 0.55, 0.9];
            }
        }
    }
    // the sharpened density, scaled to total 1 by a sum over the picture's range
    let q = |x: f32| gauss(x, -1.0, 1.0) * class_posterior(x, -1.0, 1.0, 1.0, 0.5).powf(w);
    let total: f32 = (0..10_000).map(|i| q(lo + (hi - lo) * (i as f32 + 0.5) / 10_000.0)).sum::<f32>() * (hi - lo) / 10_000.0;
    for x in 0..width {
        let v = lo + (hi - lo) * (x as f32 + 0.5) / width as f32;
        img[(x, h - 1 - to_px(gauss(v, -1.0, 1.0)))] = [0.0, 0.0, 0.0];
        img[(x, h - 1 - to_px(q(v) / total))] = [0.85, 0.1, 0.1];
    }
    img
}

fn mean_sd(v: &[f32]) -> (f32, f32) {
    let m = v.iter().sum::<f32>() / v.len() as f32;
    (m, (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt())
}

pub fn main() {
    let dir = "out/guidance";
    let levels: Vec<f32> = (0..=STEPS).map(|k| cosine_alpha_bar(1.0 - k as f32 / STEPS as f32)).collect();
    let class_a = [(1.0, -1.0, 1.0)];
    let both = [(0.5, -1.0, 1.0), (0.5, 1.0, 1.0)];
    let line_1pc = -1.0 - 2.326; // 1 % of class A lies below this
    let mut rng = Rng::new(1);
    let starts: Vec<f32> = (0..2000).map(|_| rng.normal()).collect();

    println!("class A = N(−1, 1); 1 % of its members lie below {line_1pc:.2}");
    for w in [0.0, 1.0, 3.0, 8.0] {
        let (sm, ss) = sharpened_mean_sd(w, -1.0, 1.0, 1.0, 0.5);
        println!("w = {w}: sharpened density mean {sm:5.2}, sd {ss:4.2}");
        for stochastic in [false, true] {
            let mut srng = Rng::new(2);
            let s: Vec<f32> = starts
                .iter()
                .map(|&x| guided_sample(x, &levels, w, |x, ab| eps_mixture(x, ab, &class_a), |x, ab| eps_mixture(x, ab, &both), if stochastic { Some(&mut srng) } else { None }))
                .collect();
            let (m, sd) = mean_sd(&s);
            let beyond = s.iter().filter(|&&v| v < line_1pc).count() as f32 / s.len() as f32;
            let kind = if stochastic { "stochastic   " } else { "deterministic" };
            println!("    {kind}: samples mean {m:5.2}, sd {sd:4.2}, {:5.1} % beyond the 1 % line", 100.0 * beyond);
            save_rgb(&histogram(&s, w), format!("{dir}/w{w}-{}.png", if stochastic { "stochastic" } else { "deterministic" }));
        }
    }

    // A class with a sub-mode near the other class: A = 0.6 N(−2, 0.5²) + 0.4 N(0.6, 0.5²),
    // B = N(1.5, 0.5²), equally likely.
    let a2 = [(0.6, -2.0, 0.5), (0.4, 0.6, 0.5)];
    let both2 = [(0.3, -2.0, 0.5), (0.2, 0.6, 0.5), (0.5, 1.5, 0.5)];
    println!("class A with a sub-mode at 0.6 (40 % of A), class B at 1.5:");
    for w in [0.0, 1.0, 3.0] {
        let mut srng = Rng::new(3);
        let s: Vec<f32> = starts.iter().map(|&x| guided_sample(x, &levels, w, |x, ab| eps_mixture(x, ab, &a2), |x, ab| eps_mixture(x, ab, &both2), Some(&mut srng))).collect();
        let upper = s.iter().filter(|&&v| v > -0.5).count() as f32 / s.len() as f32;
        println!("    w = {w}: {:4.1} % of the samples in the sub-mode (above −0.5)", 100.0 * upper);
    }
    println!("wrote the results to {dir}/");
}
