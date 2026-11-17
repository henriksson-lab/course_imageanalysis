// Coding exercises for "Diffusion models and generated detail". Replace each `todo!()` with your code,
// then check it with `cargo test -p diffusion` and look at `cargo run -p diffusion`.
#![allow(unused_variables, unused_imports)]

// The data are points (vectors of any length, here 1 or 2 numbers) drawn from a mixture
// of round Gaussian blobs, equally likely, with centres `centres` and standard deviation
// `spread`. For such data the best denoiser is known exactly, so no network is needed.

use imgcourse::Rng;

/// C1a. The cosine noise schedule: the signal factor ᾱ at time τ (0 = data, 1 = noise),
/// `ᾱ(τ) = cos²((τ + s)/(1 + s) · π/2) / cos²(s/(1 + s) · π/2)` with `s = 0.008`.
///
/// Hint: `std::f32::consts::FRAC_PI_2` is π/2; `.clamp(0.0, 1.0)` guards against rounding.
pub fn cosine_alpha_bar(tau: f32) -> f32 {
    todo!("C1a: cosine_alpha_bar")
}

/// C1b. The forward process in one go: `x = √ᾱ x0 + √(1 − ᾱ) ε`.
pub fn forward(x0: &[f32], alpha_bar: f32, eps: &[f32]) -> Vec<f32> {
    todo!("C1b: forward")
}

/// C2. The best denoiser: the average of all clean points compatible with the noisy point
/// `x`, `E[x0 | x]`. With `a = √ᾱ`, `b² = 1 − ᾱ`, blob k seen through the noise is a
/// round Gaussian around `a · centre_k` with variance `v = a² spread² + b²`:
/// - responsibility of blob k: `r_k ∝ exp(−|x − a centre_k|² / (2v))`, scaled to sum 1;
/// - best guess within blob k: `centre_k + (a spread² / v) (x − a centre_k)`;
/// - the answer: `Σ_k r_k · (best guess within blob k)`.
///
/// Hint: first compute the exponent `−|x − a centre_k|² / (2v)` for every blob, subtract the
/// largest of them before `exp` (very far points would otherwise give 0 / 0), then divide by
/// the sum.
pub fn best_x0(x: &[f32], alpha_bar: f32, centres: &[Vec<f32>], spread: f32) -> Vec<f32> {
    todo!("C2: best_x0")
}

/// C3. Deterministic sampling (DDIM): start at `x` (pure noise) with `alpha_bars[0]`, and go
/// through the signal levels `alpha_bars` (rising, ending at 1). At each step: estimate the
/// clean point `x̂0`, the noise `ε̂ = (x − √ᾱ x̂0) / √(1 − ᾱ)`, and move to the next level with
/// the same noise direction: `x = √ᾱ_next x̂0 + √(1 − ᾱ_next) ε̂`.
///
/// Hint: a loop over `k in 0..alpha_bars.len() - 1` with `ab = alpha_bars[k]`, `next =
/// alpha_bars[k + 1]`; `best_x0` and `forward` do most of the work.
pub fn sample_ddim(x: &[f32], alpha_bars: &[f32], centres: &[Vec<f32>], spread: f32) -> Vec<f32> {
    todo!("C3: sample_ddim")
}

/// C4. Stochastic sampling (DDPM): as C3, but each step draws the next point at random from
/// the Gaussian of the lecture's math slide. With the current level `ab`, the next `next`,
/// `α = ab / next` and `β = 1 − α`:
/// mean `(√next · β / (1 − ab)) x̂0 + (√α · (1 − next) / (1 − ab)) x`,
/// variance `β (1 − next) / (1 − ab)`.
///
/// Hint: the same loop as C3; add `sd * rng.normal()` to every coordinate of the mean.
pub fn sample_ddpm(x: &[f32], alpha_bars: &[f32], centres: &[Vec<f32>], spread: f32, rng: &mut Rng) -> Vec<f32> {
    todo!("C4: sample_ddpm")
}

/// C5. Restoration of one value with a Gaussian prior `x ~ N(mu, s²)` from a measurement
/// `y = x + noise`, noise sd `sigma`: the posterior is Gaussian with
/// mean `mu + s²/(s² + sigma²) (y − mu)` and variance `s² sigma² / (s² + sigma²)`.
/// Returns `(mean, variance)`.
pub fn posterior(mu: f32, s: f32, sigma: f32, y: f32) -> (f32, f32) {
    todo!("C5: posterior")
}
