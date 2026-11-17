// One attention layer on 8 × 8 patches of the micrograph (position only, content only, and
// non-local means on a noisy copy), and the calibration of a simulated classifier.
// Pictures in out/attention/ (enlarged 4 times).
//   cargo run -p moderndl -- attention                      (your code)
//   cargo run -p moderndl --features solution -- attention  (the model solution)

use crate::attention::*;
use imgcourse::*;

fn enlarge(img: &Image<f32>) -> Image<f32> {
    Image::from_fn(img.width() * 4, img.height() * 4, |x, y| img[(x / 4, y / 4)])
}

/// The weights of one query patch, from the same scores as `patch_attention`.
fn query_weights(img: &Image<f32>, p: usize, content: f32, position: f32, temperature: f32, q: usize) -> Vec<f32> {
    let patches = to_patches(img, p);
    let per_row = img.width() / p;
    let scores: Vec<f32> = (0..patches.len())
        .map(|j| {
            let diff: f32 = (0..p * p).map(|k| (patches[q][k] - patches[j][k]).powi(2)).sum::<f32>() / (p * p) as f32;
            let grid = ((q % per_row) as f32 - (j % per_row) as f32).powi(2) + ((q / per_row) as f32 - (j / per_row) as f32).powi(2);
            (-content * diff - position * grid) / temperature
        })
        .collect();
    weights_and_average(&scores, &patches).0
}

fn mse(a: &Image<f32>, b: &Image<f32>) -> f32 {
    a.zip_map(b, |u, v| (u - v).powi(2)).mean()
}

pub fn main() {
    let dir = "out/attention";
    let p = 8;
    let img = samples::cell().crop(64, 96, 64, 64);
    save_gray(&enlarge(&img), format!("{dir}/input.png"));
    let noisy = samples::add_noise(&img, 0.1, 2);
    save_gray(&enlarge(&noisy), format!("{dir}/noisy.png"));

    let q = 3 * 8 + 3; // the query patch at grid position (3, 3)
    println!("{:<34} {:>11} {:>11} {:>10}", "attention layer", "self weight", "effective", "error");
    for (name, input, content, position, t) in [
        ("position-only", &img, 0.0, 1.0, 1.0),
        ("content-only", &img, 50.0, 0.0, 1.0),
        ("nonlocal-means", &noisy, 50.0, 0.0, 1.0),
        ("nonlocal-means-cold", &noisy, 50.0, 0.0, 0.2),
        ("nonlocal-means-hot", &noisy, 50.0, 0.0, 5.0),
    ] {
        let out = patch_attention(input, p, content, position, t);
        let w = query_weights(input, p, content, position, t, q);
        let effective = 1.0 / w.iter().map(|v| v * v).sum::<f32>(); // how many patches it averages, in effect
        println!("{:<34} {:>10.0}% {effective:>11.1} {:>10.5}", format!("{name} (T = {t})"), 100.0 * w[q], mse(&out, &img));
        save_gray(&enlarge(&out), format!("{dir}/{name}.png"));
    }
    println!("(error: mean squared difference from the clean image; the noisy input has {:.5})", mse(&noisy, &img));

    // Calibration: two classes, x ~ N(+1, 1) or N(−1, 1); the calibrated logit is 2x.
    // The "network" is overconfident: it reports 3 · 2x.
    let sample = |n: usize, mean: f32, seed: u64| {
        let mut rng = Rng::new(seed);
        let mut logits = Vec::new();
        let mut labels = Vec::new();
        for _ in 0..n {
            let y = rng.uniform() < 0.5;
            let x = if y { mean } else { -mean } + rng.normal();
            logits.push(3.0 * 2.0 * x);
            labels.push(y);
        }
        (logits, labels)
    };
    let summary = |name: &str, logits: &[f32], labels: &[bool], t: f32| {
        let conf: Vec<f32> = logits.iter().map(|z| sigmoid((z / t).abs())).collect();
        let correct: Vec<bool> = logits.iter().zip(labels).map(|(z, &y)| (*z > 0.0) == y).collect();
        let acc = correct.iter().filter(|&&c| c).count() as f32 / correct.len() as f32;
        let mean_conf = conf.iter().sum::<f32>() / conf.len() as f32;
        println!("{name:<38} accuracy {:5.1}%  mean confidence {:5.1}%  ECE {:5.1}%", 100.0 * acc, 100.0 * mean_conf, 100.0 * ece(&conf, &correct, 10));
    };
    let (val_logits, val_labels) = sample(2000, 1.0, 1);
    let (test_logits, test_labels) = sample(2000, 1.0, 2);
    let (shift_logits, shift_labels) = sample(2000, 0.5, 3);
    let candidates: Vec<f32> = (1..=100).map(|k| 0.1 * k as f32).collect();
    let t = fit_temperature(&val_logits, &val_labels, &candidates);
    println!("\ncalibration (temperature fitted on a validation set: T = {t:.1})");
    summary("overconfident network", &test_logits, &test_labels, 1.0);
    summary(&format!("with T = {t:.1}"), &test_logits, &test_labels, t);
    summary(&format!("shifted classes, same T = {t:.1}"), &shift_logits, &shift_labels, t);
    println!("wrote the results to {dir}/");
}
