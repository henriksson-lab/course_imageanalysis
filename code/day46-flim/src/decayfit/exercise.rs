// Coding exercises for "Fitting decays with few photons". Replace each `todo!()` with your code,
// then check it with `cargo test -p day46 decayfit::` and look at `cargo run -p day46 -- decayfit`.
#![allow(unused_variables, unused_imports)]

use imgcourse::linalg;

/// The three costs of the lecture (slide "Which cost?").
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cost {
    /// Σ (y − μ)²
    LeastSquares,
    /// χ² with data weights: Σ (y − μ)² / y (a bin with 0 counts gets the weight 1)
    Neyman,
    /// The Poisson deviance 2 Σ [y log(y/μ) − (y − μ)], with 0 log 0 = 0
    Poisson,
}

/// A fitted single-exponential model: lifetime (ns), photons in the decay, background per bin.
#[derive(Clone, Copy, Debug)]
pub struct Fit {
    pub tau: f32,
    pub photons: f32,
    pub background: f32,
}

/// C1. The expected histogram (one value per bin, `irf.len()` bins over one `period`):
/// `photons` × (IRF ∗ periodic decay) + `background` in every bin. Reconvolution.
///
/// - The periodic decay (all earlier pulses summed) puts the fraction
///   (e^(−a/τ) − e^(−b/τ)) / (1 − e^(−T/τ)) of the photons in the bin from a to b.
/// - The IRF is a histogram with the same bins, summing to 1; a photon that would arrive in bin
///   i arrives in bin (i + k) mod n with probability irf[k]: a circular convolution.
///
/// Hint: first a `Vec` with the decay's fraction in each bin (bin i runs from i·Δt to (i+1)·Δt),
/// then start from `vec![background; n]` and add `photons * irf[k] * decay[i]` to bin `(i + k) % n`.
pub fn model(tau: f32, photons: f32, background: f32, irf: &[f32], period: f32) -> Vec<f32> {
    todo!("C1: model")
}

/// C2. The cost of the model `mu` for the counts `y`.
///
/// Hint: `match kind { Cost::LeastSquares => ..., Cost::Neyman => ..., Cost::Poisson => ... }`
/// for each bin; in the Poisson case use 0 for y log(y/μ) when y = 0.
pub fn cost(y: &[f32], mu: &[f32], kind: Cost) -> f32 {
    todo!("C2: cost")
}

/// C3. Fit one lifetime, its photons and a background through the forward model (Gauss–Newton
/// with step halving, the slide "Nonlinear fitting in words"). Weights: 1 for least squares,
/// 1/y for Neyman (1 for empty bins), 1/μ for Poisson (re-evaluated every step).
///
/// Hint: start from τ = `start_tau`, photons = 0.9 · total, background = 0.1 · total / bins.
/// In each step: the model μ; the three columns of the Jacobian (∂μ/∂τ by a small finite
/// difference, ∂μ/∂photons = (μ − background) / photons, ∂μ/∂background = 1); the 3 × 3 normal
/// equations JᵀWJ δ = JᵀW(y − μ), solved with `linalg::solve` (in f64). If δ would make the
/// background negative, fix its change at −background (move that column to the right-hand
/// side) and solve the 2 × 2 system for τ and photons alone. Then try the step δ, δ/2, δ/4, ...
/// until the cost goes down (keep τ > 0, photons > 0, background ≥ 0). Stop after 50 steps, or
/// when the cost no longer goes down.
pub fn fit(y: &[f32], irf: &[f32], period: f32, kind: Cost, start_tau: f32) -> Fit {
    todo!("C3: fit")
}

/// C4. The two mean lifetimes of a mixture with amplitude (molecule) fractions `alpha` and
/// lifetimes `tau`: (amplitude-weighted τ̄a = Σ αᵢτᵢ / Σ αᵢ, intensity-weighted τ̄f = Σ fᵢτᵢ),
/// with the photon fractions fᵢ = αᵢτᵢ / Σ αⱼτⱼ.
pub fn mean_lifetimes(alpha: &[f32], tau: &[f32]) -> (f32, f32) {
    todo!("C4: mean_lifetimes")
}

/// C5. Global analysis, per pixel: the amplitudes a (photons of each species) that make
/// Σₖ aₖ shapesₖ closest to the counts y in the least-squares sense (the shapes are the expected
/// histograms of one photon of each species, or any known columns). Linear: no search.
///
/// Hint: the matrix has one row per bin and one column per shape; `linalg::least_squares(&m, &b)`
/// solves it (rows as `Vec<f64>`).
pub fn amplitudes(y: &[f32], shapes: &[Vec<f32>]) -> Vec<f32> {
    todo!("C5: amplitudes")
}
