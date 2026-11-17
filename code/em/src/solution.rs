// Model solution for the coding exercises of "Fitting a mixture: the EM algorithm".

/// A mixture of `K` Gaussians in 1D: weights `π_k`, means `μ_k` and variances `σ_k²`.
#[derive(Clone, Debug)]
pub struct Mixture {
    pub weights: Vec<f32>,
    pub means: Vec<f32>,
    pub variances: Vec<f32>,
}

/// The normal density `N(x | μ, σ²)`. (Given.)
pub fn normal(x: f32, mean: f32, variance: f32) -> f32 {
    (-(x - mean) * (x - mean) / (2.0 * variance)).exp() / (2.0 * std::f32::consts::PI * variance).sqrt()
}

/// C1. The E-step (slide "The EM algorithm"): the responsibilities `r[i][k]` of every component
/// for every data point, and the log-likelihood `Σ_i log p(x_i)` of the current mixture.
pub fn e_step(data: &[f32], m: &Mixture) -> (Vec<Vec<f32>>, f32) {
    let mut r = Vec::new();
    let mut log_likelihood = 0.0;
    for &x in data {
        let parts: Vec<f32> = (0..m.weights.len()).map(|k| m.weights[k] * normal(x, m.means[k], m.variances[k])).collect();
        let p: f32 = parts.iter().sum();
        log_likelihood += p.ln();
        r.push(parts.iter().map(|q| q / p).collect());
    }
    (r, log_likelihood)
}

/// C2. The M-step: refit every component with the responsibilities as weights:
/// `N_k = Σ_i r_ik`, `π_k = N_k / n`, `μ_k = Σ_i r_ik x_i / N_k`,
/// `σ_k² = Σ_i r_ik (x_i − μ_k)² / N_k`, and then at least `floor` (slide "Collapse").
pub fn m_step(data: &[f32], r: &[Vec<f32>], floor: f32) -> Mixture {
    let k_count = r[0].len();
    let n = data.len() as f32;
    let mut m = Mixture { weights: vec![0.0; k_count], means: vec![0.0; k_count], variances: vec![0.0; k_count] };
    for k in 0..k_count {
        let mut nk = 0.0;
        let mut sum = 0.0;
        for i in 0..data.len() {
            nk += r[i][k];
            sum += r[i][k] * data[i];
        }
        let mean = sum / nk;
        let mut var = 0.0;
        for i in 0..data.len() {
            var += r[i][k] * (data[i] - mean) * (data[i] - mean);
        }
        m.weights[k] = nk / n;
        m.means[k] = mean;
        m.variances[k] = (var / nk).max(floor);
    }
    m
}

/// C3. EM from a starting mixture: E-step, M-step, repeat, until the log-likelihood rises by
/// less than `tol` (or after `max_iterations`). Returns the final mixture and the
/// log-likelihood of every E-step (the first one is that of the start).
pub fn em(data: &[f32], start: &Mixture, floor: f32, tol: f32, max_iterations: usize) -> (Mixture, Vec<f32>) {
    let mut m = start.clone();
    let mut trace: Vec<f32> = Vec::new();
    for _ in 0..max_iterations {
        let (r, ll) = e_step(data, &m);
        if let Some(&last) = trace.last() {
            if ll - last < tol {
                trace.push(ll);
                break;
            }
        }
        trace.push(ll);
        m = m_step(data, &r, floor);
    }
    (m, trace)
}

/// The number of parameters of a 1D mixture with `k` components: `k − 1` weights, `k` means,
/// `k` variances. (Given.)
pub fn parameters(k: usize) -> usize {
    3 * k - 1
}

/// C4. The information criteria (slide "How many components?"): `BIC = −2 log p + p ln n`
/// and `AIC = −2 log p + 2p`, with `p` parameters and `n` data points. Returns `(BIC, AIC)`.
/// Smaller is better.
pub fn bic_aic(log_likelihood: f32, p: usize, n: usize) -> (f32, f32) {
    let bic = -2.0 * log_likelihood + p as f32 * (n as f32).ln();
    let aic = -2.0 * log_likelihood + 2.0 * p as f32;
    (bic, aic)
}
