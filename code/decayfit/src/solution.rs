// Model solution for the coding exercises of "Fitting decays with few photons".

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
pub fn model(tau: f32, photons: f32, background: f32, irf: &[f32], period: f32) -> Vec<f32> {
    let n = irf.len();
    let dt = period / n as f32;
    let norm = 1.0 - (-period / tau).exp();
    let decay: Vec<f32> = (0..n).map(|i| ((-(i as f32) * dt / tau).exp() - (-((i + 1) as f32) * dt / tau).exp()) / norm).collect();
    let mut mu = vec![background; n];
    for k in 0..n {
        if irf[k] == 0.0 {
            continue;
        }
        for i in 0..n {
            mu[(i + k) % n] += photons * irf[k] * decay[i];
        }
    }
    mu
}

/// C2. The cost of the model `mu` for the counts `y`.
pub fn cost(y: &[f32], mu: &[f32], kind: Cost) -> f32 {
    let mut sum = 0.0;
    for (&y, &m) in y.iter().zip(mu) {
        sum += match kind {
            Cost::LeastSquares => (y - m) * (y - m),
            Cost::Neyman => (y - m) * (y - m) / y.max(1.0),
            Cost::Poisson => {
                let ylogy = if y > 0.0 { y * (y / m).ln() } else { 0.0 };
                2.0 * (ylogy - (y - m))
            }
        };
    }
    sum
}

/// C3. Fit one lifetime, its photons and a background through the forward model (Gauss–Newton
/// with step halving, the slide "Nonlinear fitting in words"). Weights: 1 for least squares,
/// 1/y for Neyman (1 for empty bins), 1/μ for Poisson (re-evaluated every step).
pub fn fit(y: &[f32], irf: &[f32], period: f32, kind: Cost, start_tau: f32) -> Fit {
    let total: f32 = y.iter().sum();
    let mut p = [start_tau, 0.9 * total, 0.1 * total / y.len() as f32]; // τ, photons, background
    let mut current = cost(y, &model(p[0], p[1], p[2], irf, period), kind);
    for _ in 0..50 {
        let mu = model(p[0], p[1], p[2], irf, period);
        // Jacobian: τ by a finite difference; photons and background exactly (the model is linear in them)
        let h = 1e-3 * p[0];
        let mu_h = model(p[0] + h, p[1], p[2], irf, period);
        let jac: Vec<[f64; 3]> = (0..y.len())
            .map(|j| [((mu_h[j] - mu[j]) / h) as f64, ((mu[j] - p[2]) / p[1]) as f64, 1.0])
            .collect();
        // normal equations JᵀWJ δ = JᵀW (y − μ)
        let mut a = vec![vec![0.0f64; 3]; 3];
        let mut b = vec![0.0f64; 3];
        for j in 0..y.len() {
            let w = match kind {
                Cost::LeastSquares => 1.0,
                Cost::Neyman => 1.0 / y[j].max(1.0) as f64,
                Cost::Poisson => 1.0 / mu[j].max(1e-6) as f64,
            };
            for r in 0..3 {
                b[r] += w * jac[j][r] * (y[j] - mu[j]) as f64;
                for c in 0..3 {
                    a[r][c] += w * jac[j][r] * jac[j][c];
                }
            }
        }
        let Some(mut delta) = linalg::solve(&a, &b) else { break };
        if p[2] + (delta[2] as f32) < 0.0 {
            // The background would become negative: set it to 0 and solve for τ and photons only.
            let db = -p[2] as f64;
            let a2 = vec![vec![a[0][0], a[0][1]], vec![a[1][0], a[1][1]]];
            let b2 = vec![b[0] - a[0][2] * db, b[1] - a[1][2] * db];
            let Some(d2) = linalg::solve(&a2, &b2) else { break };
            delta = vec![d2[0], d2[1], db];
        }
        // step; halve it while the cost does not go down
        let mut step = 1.0;
        let mut improved = false;
        for _ in 0..20 {
            let q = [
                (p[0] + step * delta[0] as f32).max(0.01),
                (p[1] + step * delta[1] as f32).max(1.0),
                (p[2] + step * delta[2] as f32).max(0.0),
            ];
            let c = cost(y, &model(q[0], q[1], q[2], irf, period), kind);
            if c < current {
                improved = current - c > 1e-6 * current.abs();
                p = q;
                current = c;
                break;
            }
            step *= 0.5;
        }
        if !improved {
            break;
        }
    }
    Fit { tau: p[0], photons: p[1], background: p[2] }
}

/// C4. The two mean lifetimes of a mixture with amplitude (molecule) fractions `alpha` and
/// lifetimes `tau`: (amplitude-weighted τ̄a = Σ αᵢτᵢ / Σ αᵢ, intensity-weighted τ̄f = Σ fᵢτᵢ),
/// with the photon fractions fᵢ = αᵢτᵢ / Σ αⱼτⱼ.
pub fn mean_lifetimes(alpha: &[f32], tau: &[f32]) -> (f32, f32) {
    let (mut sa, mut sat, mut satt) = (0.0, 0.0, 0.0);
    for (&a, &t) in alpha.iter().zip(tau) {
        sa += a;
        sat += a * t;
        satt += a * t * t;
    }
    (sat / sa, satt / sat)
}

/// C5. Global analysis, per pixel: the amplitudes a (photons of each species) that make
/// Σₖ aₖ shapesₖ closest to the counts y in the least-squares sense (the shapes are the expected
/// histograms of one photon of each species, or any known columns). Linear: no search.
pub fn amplitudes(y: &[f32], shapes: &[Vec<f32>]) -> Vec<f32> {
    let m: Vec<Vec<f64>> = (0..y.len()).map(|j| shapes.iter().map(|s| s[j] as f64).collect()).collect();
    let b: Vec<f64> = y.iter().map(|&v| v as f64).collect();
    match linalg::least_squares(&m, &b) {
        Some(a) => a.iter().map(|&v| v as f32).collect(),
        None => vec![0.0; shapes.len()],
    }
}
