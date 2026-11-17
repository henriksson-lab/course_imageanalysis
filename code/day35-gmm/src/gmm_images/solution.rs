// Model solution for the coding exercises of "Mixtures on images".

use imgcourse::Image;

/// A mixture of `K` Gaussians over grey levels 0 to 255: weights `π_k`, means `μ_k` and
/// variances `σ_k²`.
#[derive(Clone, Debug)]
pub struct Mixture {
    pub weights: Vec<f32>,
    pub means: Vec<f32>,
    pub variances: Vec<f32>,
}

/// `log N(g | μ, σ²)`, the logarithm of the normal density. (Given.)
pub fn log_normal(g: f32, mean: f32, variance: f32) -> f32 {
    -(g - mean) * (g - mean) / (2.0 * variance) - 0.5 * (2.0 * std::f32::consts::PI * variance).ln()
}

/// C1. The histogram of an image with grey values in [0, 1]: 256 bins, bin `j` counts the
/// pixels whose grey level `round(255 v)` is `j`.
pub fn histogram(img: &Image<f32>) -> Vec<f32> {
    let mut h = vec![0.0; 256];
    for &v in img.pixels() {
        let j = (255.0 * v).round().clamp(0.0, 255.0) as usize;
        h[j] += 1.0;
    }
    h
}

/// C2. `log Σ_k e^(a_k)`, computed safely as `a_max + log Σ_k e^(a_k − a_max)` (slide "The
/// math: EM on a weighted histogram"), so that very negative `a_k` do not give `log 0`.
pub fn log_sum_exp(a: &[f32]) -> f32 {
    let max = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = a.iter().map(|x| (x - max).exp()).sum();
    max + sum.ln()
}

/// C3. The posteriors (responsibilities) of the classes for the grey level `g`:
/// `r_k = π_k N(g | μ_k, σ_k²) / Σ_j π_j N(g | μ_j, σ_j²)`, computed with logarithms:
/// `a_k = log π_k + log N(g | μ_k, σ_k²)`, `r_k = e^(a_k − log_sum_exp(a))`.
pub fn posteriors(g: f32, m: &Mixture) -> Vec<f32> {
    let a: Vec<f32> = (0..m.weights.len()).map(|k| m.weights[k].ln() + log_normal(g, m.means[k], m.variances[k])).collect();
    let total = log_sum_exp(&a);
    a.iter().map(|ak| (ak - total).exp()).collect()
}

/// C4. EM on a histogram (slide "The math: EM on a weighted histogram"): every bin `j` is one
/// point, the grey level `j`, with weight `hist[j]`. Runs `iterations` E- and M-steps from
/// `start`, with variances at least `floor`. Returns the mixture and the log-likelihood
/// `Σ_j hist[j] · log p(j)` of every E-step.
pub fn em_histogram(hist: &[f32], start: &Mixture, floor: f32, iterations: usize) -> (Mixture, Vec<f32>) {
    let kk = start.weights.len();
    let n: f32 = hist.iter().sum();
    let mut m = start.clone();
    let mut trace = Vec::new();
    for _ in 0..iterations {
        // E-step: the responsibilities r[j][k] of every bin, and the log-likelihood
        let mut r = vec![vec![0.0; kk]; hist.len()];
        let mut ll = 0.0;
        for j in 0..hist.len() {
            let g = j as f32;
            let a: Vec<f32> = (0..kk).map(|k| m.weights[k].ln() + log_normal(g, m.means[k], m.variances[k])).collect();
            let total = log_sum_exp(&a);
            ll += hist[j] * total;
            for k in 0..kk {
                r[j][k] = (a[k] - total).exp();
            }
        }
        trace.push(ll);
        // M-step: the sums over pixels become sums over bins, weighted by the counts
        for k in 0..kk {
            let (mut nk, mut sum) = (0.0, 0.0);
            for j in 0..hist.len() {
                nk += hist[j] * r[j][k];
                sum += hist[j] * r[j][k] * j as f32;
            }
            let mean = sum / nk;
            let mut var = 0.0;
            for j in 0..hist.len() {
                var += hist[j] * r[j][k] * (j as f32 - mean) * (j as f32 - mean);
            }
            m.weights[k] = nk / n;
            m.means[k] = mean;
            m.variances[k] = (var / nk).max(floor);
        }
    }
    (m, trace)
}

/// C5a. The label image: every pixel gets the class (1, 2, ..., K) with the largest posterior
/// at its grey level `255 v`.
pub fn label_image(img: &Image<f32>, m: &Mixture) -> Image<u32> {
    img.map(|v| {
        let r = posteriors(255.0 * v, m);
        let mut best = 0;
        for k in 1..r.len() {
            if r[k] > r[best] {
                best = k;
            }
        }
        best as u32 + 1
    })
}

/// C5b. The probability map of class `k` (counted from 0): its posterior at every pixel.
pub fn probability_map(img: &Image<f32>, m: &Mixture, k: usize) -> Image<f32> {
    img.map(|v| posteriors(255.0 * v, m)[k])
}
