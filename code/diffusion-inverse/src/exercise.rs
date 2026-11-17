// Coding exercises for "Diffusion priors for inverse problems, and where detail comes from". Replace each `todo!()` with your code,
// then check it with `cargo test -p diffusion-inverse` and look at `cargo run -p diffusion-inverse`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Rng;

/// Cosine number `k` for signals of length `n`, scaled to length 1 (the orthonormal DCT
/// basis): `c_k(i) = s_k cos(π k (i + ½) / n)` with `s_0 = √(1/n)` and `s_k = √(2/n)`.
pub fn cosine(k: usize, n: usize) -> Vec<f32> {
    let s = if k == 0 { (1.0 / n as f32).sqrt() } else { (2.0 / n as f32).sqrt() };
    (0..n).map(|i| s * (std::f32::consts::PI * k as f32 * (i as f32 + 0.5) / n as f32).cos()).collect()
}

/// The measurement of an ideal low-pass that keeps the first `keep` cosines: the
/// coefficients `<x, c_k>` for `k = 0 ... keep − 1` (without noise).
pub fn measure(x: &[f32], keep: usize) -> Vec<f32> {
    (0..keep).map(|k| cosine(k, x.len()).iter().zip(x).map(|(c, v)| c * v).sum()).collect()
}

/// C1. Split a signal into the part the low-pass sees and the part in its null space:
/// `seen = Σ_{k < keep} <x, c_k> c_k` and `unseen = x − seen`. Returns `(seen, unseen)`.
///
/// Hint: `measure(x, keep)` gives the coefficients `<x, c_k>`.
pub fn split_measured(x: &[f32], keep: usize) -> (Vec<f32>, Vec<f32>) {
    todo!("C1: split_measured")
}

/// C2. The posterior weights of the training signals, given the measurement `y` and the
/// measurement of every training signal (`measured[j]`), the prior weights `prior[j]` and the
/// noise sd `sigma_n`: `w_j ∝ prior_j · exp(−|y − measured_j|² / (2 σ_n²))`, scaled to sum 1.
///
/// Hint: subtract the largest exponent before taking `exp`, or every weight may come out 0.
pub fn posterior_weights(y: &[f32], measured: &[Vec<f32>], prior: &[f32], sigma_n: f32) -> Vec<f32> {
    todo!("C2: posterior_weights")
}

/// C3. The posterior mean: `Σ_j w_j x_j`.
pub fn posterior_mean(weights: &[f32], signals: &[Vec<f32>]) -> Vec<f32> {
    todo!("C3: posterior_mean")
}

/// C4. A posterior sample: draw the index `j` with probability `w_j` (a uniform number
/// `u` in [0, 1), then the first `j` whose running sum of weights exceeds `u`).
pub fn posterior_sample(weights: &[f32], rng: &mut Rng) -> usize {
    todo!("C4: posterior_sample")
}

/// C5. Coverage: the fraction of positions `i` at which `truth[i]` lies inside the 90 %
/// band of the samples, from their 5th to their 95th percentile at that position.
///
/// Hint: for `m` sorted values, take the values at index `(0.05 · m) as usize` and
/// `((0.95 · m) as usize).min(m − 1)` as the ends of the band.
pub fn band_coverage(truth: &[f32], samples: &[Vec<f32>]) -> f32 {
    todo!("C5: band_coverage")
}
