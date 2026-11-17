// Coding exercises for "Fitting a mixture: the EM algorithm". Replace each `todo!()` with your
// code, then check it with `cargo test -p em` and look at `cargo run -p em`.
#![allow(unused_variables, unused_imports)]

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
///
/// Hint: use `normal`; `p.ln()` is the natural logarithm.
pub fn e_step(data: &[f32], m: &Mixture) -> (Vec<Vec<f32>>, f32) {
    todo!("C1: e_step")
}

/// C2. The M-step: refit every component with the responsibilities as weights:
/// `N_k = Σ_i r_ik`, `π_k = N_k / n`, `μ_k = Σ_i r_ik x_i / N_k`,
/// `σ_k² = Σ_i r_ik (x_i − μ_k)² / N_k`, and then at least `floor` (slide "Collapse").
pub fn m_step(data: &[f32], r: &[Vec<f32>], floor: f32) -> Mixture {
    todo!("C2: m_step")
}

/// C3. EM from a starting mixture: E-step, M-step, repeat, until the log-likelihood rises by
/// less than `tol` (or after `max_iterations`). Returns the final mixture and the
/// log-likelihood of every E-step (the first one is that of the start).
///
/// Hint: compute the E-step first; stop (after recording its log-likelihood) when it rose by less
/// than `tol` since the previous one; otherwise do the M-step.
pub fn em(data: &[f32], start: &Mixture, floor: f32, tol: f32, max_iterations: usize) -> (Mixture, Vec<f32>) {
    todo!("C3: em")
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
    todo!("C4: bic_aic")
}
