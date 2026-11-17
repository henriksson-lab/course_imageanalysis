// Fits grey-level mixtures by EM on the histogram of the course micrograph and of a
// synthetic fluorescence image; label images and probability maps in out/gmm-images/.
//   cargo run -p day35 -- gmm-images                      (your code)
//   cargo run -p day35 --features solution -- gmm-images  (the model solution)

use crate::gmm_images::*;
use imgcourse::*;

/// A start for K classes: means spread evenly between the 5th and 95th percentile of the
/// histogram, equal weights, wide variances.
fn start(hist: &[f32], k: usize) -> Mixture {
    let n: f32 = hist.iter().sum();
    let percentile = |q: f32| {
        let mut sum = 0.0;
        for (j, h) in hist.iter().enumerate() {
            sum += h;
            if sum >= q * n {
                return j as f32;
            }
        }
        255.0
    };
    let (lo, hi) = (percentile(0.05), percentile(0.95));
    let means = (0..k).map(|i| lo + (hi - lo) * (i as f32 + 0.5) / k as f32).collect();
    let sd = (hi - lo) / k as f32;
    Mixture { weights: vec![1.0 / k as f32; k], means, variances: vec![sd * sd; k] }
}

/// Sort the classes by their means, dark first, so that the labels are in grey-level order.
fn sorted(m: Mixture) -> Mixture {
    let mut order: Vec<usize> = (0..m.means.len()).collect();
    order.sort_by(|&a, &b| m.means[a].total_cmp(&m.means[b]));
    Mixture {
        weights: order.iter().map(|&k| m.weights[k]).collect(),
        means: order.iter().map(|&k| m.means[k]).collect(),
        variances: order.iter().map(|&k| m.variances[k]).collect(),
    }
}

/// The grey levels where the most probable class changes (the Bayes boundaries).
fn boundaries(m: &Mixture) -> Vec<String> {
    let label = |g: f32| {
        let r = posteriors(g, m);
        (0..r.len()).max_by(|&a, &b| r[a].total_cmp(&r[b])).unwrap()
    };
    (1..256).filter(|&g| label(g as f32) != label(g as f32 - 1.0)).map(|g| format!("{g}")).collect()
}

fn fit(name: &str, img: &Image<f32>, k: usize) -> Mixture {
    let hist = histogram(img);
    let (m, trace) = em_histogram(&hist, &start(&hist, k), 1.0, 300);
    let m = sorted(m);
    let ll = *trace.last().unwrap();
    let p = 3 * k - 1;
    let bic = -2.0 * ll + p as f32 * hist.iter().sum::<f32>().ln();
    println!("{name}, K = {k}: log p = {ll:.0}, BIC = {bic:.0}");
    for c in 0..k {
        println!("  class {}: weight {:.2}, mean {:.1}, σ {:.1}", c + 1, m.weights[c], m.means[c], m.variances[c].sqrt());
    }
    println!("  boundaries at grey levels {}", boundaries(&m).join(", "));
    m
}

pub fn main() {
    let dir = "out/gmm-images";

    // 1. The course micrograph, K = 2 and K = 3 (and BIC for K = 1 to 5).
    let cell = samples::cell();
    save_gray(&cell, format!("{dir}/cell.png"));
    for k in 1..=5 {
        let m = fit("micrograph", &cell, k);
        if k == 2 || k == 3 {
            save_labels(&label_image(&cell, &m), format!("{dir}/cell-k{k}-labels.png"));
        }
        if k == 3 {
            save_gray(&probability_map(&cell, &m, 1), format!("{dir}/cell-k3-probability-class2.png"));
        }
    }

    // 2. A synthetic fluorescence image with the true nuclei, K = 2; then with more noise.
    let (nuclei, truth) = samples::nuclei(256, 25, 3);
    for (name, img) in [("nuclei", nuclei.clone()), ("nuclei-noisy", samples::add_noise(&nuclei, 0.12, 4))] {
        let img = img.map(|v| v.clamp(0.0, 1.0));
        let m = fit(name, &img, 2);
        let labels = label_image(&img, &m);
        let correct = img.coords().filter(|&p| (labels[p] == 2) == (truth[p] > 0)).count();
        println!("  {:.1} % of the pixels as in the truth", 100.0 * correct as f32 / (256.0 * 256.0));
        save_gray(&img, format!("{dir}/{name}.png"));
        save_labels(&labels, format!("{dir}/{name}-labels.png"));
        save_gray(&probability_map(&img, &m, 1), format!("{dir}/{name}-probability.png"));
    }
    println!("wrote the pictures to {dir}/");
}
