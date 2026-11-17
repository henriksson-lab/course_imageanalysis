// Coding exercises for "Conditioning, guidance and latent diffusion". Replace each `todo!()` with your code,
// then check it with `cargo test -p guidance` and look at `cargo run -p guidance`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Rng;

/// The cosine noise schedule: the signal level ᾱ at time τ (0 = data, 1 = noise), kept at
/// least 1e-4 so that the clean estimate can always be computed.
pub fn cosine_alpha_bar(tau: f32) -> f32 {
    let s = 0.008;
    let f = |t: f32| ((t + s) / (1.0 + s) * std::f32::consts::FRAC_PI_2).cos().powi(2);
    (f(tau) / f(0.0)).clamp(1e-4, 1.0)
}

/// C1. Classifier-free guidance: `(1 + w) ε̂_cond − w ε̂_uncond`.
pub fn guided(eps_cond: f32, eps_uncond: f32, w: f32) -> f32 {
    todo!("C1: guided")
}

/// The density of a Gaussian with mean `m` and standard deviation `s` at `x`.
pub fn gauss(x: f32, m: f32, s: f32) -> f32 {
    (-(x - m).powi(2) / (2.0 * s * s)).exp() / (s * (2.0 * std::f32::consts::PI).sqrt())
}

/// C2. The class probability `p(A | x)` for two Gaussian classes with the same standard
/// deviation `sd`, A around `mean_a` with prior probability `prior_a`, B around `mean_b`
/// (Bayes: prior times density, divided by the sum over both classes).
pub fn class_posterior(x: f32, mean_a: f32, mean_b: f32, sd: f32, prior_a: f32) -> f32 {
    todo!("C2: class_posterior")
}

/// C3. The mean and standard deviation of the sharpened density of class A,
/// `p(x | A) p(A | x)^w` (scaled to total 1), by summing over a fine grid of x from
/// −10 to 10 in steps of 0.001.
///
/// Hint: sum the weights `q(x)`, `x q(x)` and `x² q(x)` over the grid; then the mean is
/// `Σ x q / Σ q` and the variance `Σ x² q / Σ q − mean²`.
pub fn sharpened_mean_sd(w: f32, mean_a: f32, mean_b: f32, sd: f32, prior_a: f32) -> (f32, f32) {
    todo!("C3: sharpened_mean_sd")
}

/// The exact noise prediction `ε̂(x, ᾱ)` for data from a mixture of Gaussians, given as
/// `(weight, mean, sd)`: the best clean estimate `x̂0 = E[x0 | x]`, then `ε̂ = (x − √ᾱ x̂0) / √(1 − ᾱ)`.
/// This plays the role of the trained network.
pub fn eps_mixture(x: f32, alpha_bar: f32, comps: &[(f32, f32, f32)]) -> f32 {
    let a = alpha_bar.sqrt();
    let (mut num, mut den) = (0.0, 0.0);
    for &(wk, m, s) in comps {
        let v = a * a * s * s + 1.0 - alpha_bar; // variance of component k seen through the noise
        let r = wk * (-(x - a * m).powi(2) / (2.0 * v)).exp() / v.sqrt();
        num += r * (m + a * s * s / v * (x - a * m));
        den += r;
    }
    (x - a * num / den) / (1.0 - alpha_bar).sqrt()
}

/// C4. Guided sampling. Start at `x` (pure noise) with `alpha_bars[0]` and go through the
/// signal levels `alpha_bars` (rising, ending at 1). At each level ᾱ: the guided noise
/// `ε̂ = guided(eps_cond(x, ᾱ), eps_uncond(x, ᾱ), w)` and the clean estimate
/// `x̂0 = (x − √(1 − ᾱ) ε̂) / √ᾱ`. Then the step to the next level ᾱ′:
/// - deterministic (`rng` is `None`): `x = √ᾱ′ x̂0 + √(1 − ᾱ′) ε̂`;
/// - stochastic (`Some(rng)`): with `α = ᾱ / ᾱ′`, `β = 1 − α`, draw x from the Gaussian with
///   mean `(√ᾱ′ β / (1 − ᾱ)) x̂0 + (√α (1 − ᾱ′) / (1 − ᾱ)) x` and variance `β (1 − ᾱ′) / (1 − ᾱ)`.
pub fn guided_sample(x: f32, alpha_bars: &[f32], w: f32, eps_cond: impl Fn(f32, f32) -> f32, eps_uncond: impl Fn(f32, f32) -> f32, rng: Option<&mut Rng>) -> f32 {
    todo!("C4: guided_sample")
}
