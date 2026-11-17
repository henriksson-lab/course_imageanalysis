// Model solution for the coding exercises of "Diffusion priors for inverse problems, and where detail comes from".

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
pub fn split_measured(x: &[f32], keep: usize) -> (Vec<f32>, Vec<f32>) {
    let n = x.len();
    let coefficients = measure(x, keep);
    let mut seen = vec![0.0; n];
    for (k, a) in coefficients.iter().enumerate() {
        let c = cosine(k, n);
        for i in 0..n {
            seen[i] += a * c[i];
        }
    }
    let unseen = (0..n).map(|i| x[i] - seen[i]).collect();
    (seen, unseen)
}

/// C2. The posterior weights of the training signals, given the measurement `y` and the
/// measurement of every training signal (`measured[j]`), the prior weights `prior[j]` and the
/// noise sd `sigma_n`: `w_j ∝ prior_j · exp(−|y − measured_j|² / (2 σ_n²))`, scaled to sum 1.
pub fn posterior_weights(y: &[f32], measured: &[Vec<f32>], prior: &[f32], sigma_n: f32) -> Vec<f32> {
    let log_w: Vec<f32> = measured
        .iter()
        .zip(prior)
        .map(|(m, &p)| p.ln() - y.iter().zip(m).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / (2.0 * sigma_n * sigma_n))
        .collect();
    let top = log_w.iter().cloned().fold(f32::MIN, f32::max);
    let w: Vec<f32> = log_w.iter().map(|l| (l - top).exp()).collect();
    let total: f32 = w.iter().sum();
    w.iter().map(|v| v / total).collect()
}

/// C3. The posterior mean: `Σ_j w_j x_j`.
pub fn posterior_mean(weights: &[f32], signals: &[Vec<f32>]) -> Vec<f32> {
    let mut mean = vec![0.0; signals[0].len()];
    for (w, s) in weights.iter().zip(signals) {
        for i in 0..mean.len() {
            mean[i] += w * s[i];
        }
    }
    mean
}

/// C4. A posterior sample: draw the index `j` with probability `w_j` (a uniform number
/// `u` in [0, 1), then the first `j` whose running sum of weights exceeds `u`).
pub fn posterior_sample(weights: &[f32], rng: &mut Rng) -> usize {
    let u = rng.uniform();
    let mut sum = 0.0;
    for (j, w) in weights.iter().enumerate() {
        sum += w;
        if sum > u {
            return j;
        }
    }
    weights.len() - 1
}

/// C5. Coverage: the fraction of positions `i` at which `truth[i]` lies inside the 90 %
/// band of the samples, from their 5th to their 95th percentile at that position.
pub fn band_coverage(truth: &[f32], samples: &[Vec<f32>]) -> f32 {
    let m = samples.len();
    let mut inside = 0;
    for i in 0..truth.len() {
        let mut column: Vec<f32> = samples.iter().map(|s| s[i]).collect();
        column.sort_by(|a, b| a.total_cmp(b));
        let (lo, hi) = (column[(0.05 * m as f32) as usize], column[((0.95 * m as f32) as usize).min(m - 1)]);
        if lo <= truth[i] && truth[i] <= hi {
            inside += 1;
        }
    }
    inside as f32 / truth.len() as f32
}
