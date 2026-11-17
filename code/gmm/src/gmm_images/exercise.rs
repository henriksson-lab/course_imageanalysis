// Coding exercises for "Mixtures on images". Replace each `todo!()` with your
// code, then check it with `cargo test -p gmm gmm_images::` and look at `cargo run -p gmm -- gmm-images`.
#![allow(unused_variables, unused_imports)]

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
///
/// Hint: `(255.0 * v).round().clamp(0.0, 255.0) as usize` is the bin of grey value `v`.
pub fn histogram(img: &Image<f32>) -> Vec<f32> {
    todo!("C1: histogram")
}

/// C2. `log Σ_k e^(a_k)`, computed safely as `a_max + log Σ_k e^(a_k − a_max)` (slide "The
/// math: EM on a weighted histogram"), so that very negative `a_k` do not give `log 0`.
///
/// Hint: `a.iter().cloned().fold(f32::NEG_INFINITY, f32::max)` is the largest `a_k`.
pub fn log_sum_exp(a: &[f32]) -> f32 {
    todo!("C2: log_sum_exp")
}

/// C3. The posteriors (responsibilities) of the classes for the grey level `g`:
/// `r_k = π_k N(g | μ_k, σ_k²) / Σ_j π_j N(g | μ_j, σ_j²)`, computed with logarithms:
/// `a_k = log π_k + log N(g | μ_k, σ_k²)`, `r_k = e^(a_k − log_sum_exp(a))`.
///
/// Hint: use `log_normal` and `log_sum_exp`.
pub fn posteriors(g: f32, m: &Mixture) -> Vec<f32> {
    todo!("C3: posteriors")
}

/// C4. EM on a histogram (slide "The math: EM on a weighted histogram"): every bin `j` is one
/// point, the grey level `j`, with weight `hist[j]`. Runs `iterations` E- and M-steps from
/// `start`, with variances at least `floor`. Returns the mixture and the log-likelihood
/// `Σ_j hist[j] · log p(j)` of every E-step.
///
/// Hint: in every iteration, first the E-step for all 256 bins (keep `r[j][k]` in a
/// `vec![vec![0.0; K]; 256]`), then the M-step for every class, with every sum weighted by `hist[j]`.
pub fn em_histogram(hist: &[f32], start: &Mixture, floor: f32, iterations: usize) -> (Mixture, Vec<f32>) {
    todo!("C4: em_histogram")
}

/// C5a. The label image: every pixel gets the class (1, 2, ..., K) with the largest posterior
/// at its grey level `255 v`.
///
/// Hint: `img.map(|v| ...)` applies a function to every pixel.
pub fn label_image(img: &Image<f32>, m: &Mixture) -> Image<u32> {
    todo!("C5a: label_image")
}

/// C5b. The probability map of class `k` (counted from 0): its posterior at every pixel.
pub fn probability_map(img: &Image<f32>, m: &Mixture, k: usize) -> Image<f32> {
    todo!("C5b: probability_map")
}
