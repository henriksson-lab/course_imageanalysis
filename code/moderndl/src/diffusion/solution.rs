// Model solution for the coding exercises of "Diffusion models and generated detail".
//
// The data are points (vectors of any length, here 1 or 2 numbers) drawn from a mixture
// of round Gaussian blobs, equally likely, with centres `centres` and standard deviation
// `spread`. For such data the best denoiser is known exactly, so no network is needed.

use imgcourse::Rng;

/// C1a. The cosine noise schedule: the signal factor ᾱ at time τ (0 = data, 1 = noise),
/// `ᾱ(τ) = cos²((τ + s)/(1 + s) · π/2) / cos²(s/(1 + s) · π/2)` with `s = 0.008`.
pub fn cosine_alpha_bar(tau: f32) -> f32 {
    let s = 0.008;
    let f = |t: f32| ((t + s) / (1.0 + s) * std::f32::consts::FRAC_PI_2).cos().powi(2);
    (f(tau) / f(0.0)).clamp(0.0, 1.0)
}

/// C1b. The forward process in one go: `x = √ᾱ x0 + √(1 − ᾱ) ε`.
pub fn forward(x0: &[f32], alpha_bar: f32, eps: &[f32]) -> Vec<f32> {
    let (a, b) = (alpha_bar.sqrt(), (1.0 - alpha_bar).sqrt());
    (0..x0.len()).map(|i| a * x0[i] + b * eps[i]).collect()
}

/// C2. The best denoiser: the average of all clean points compatible with the noisy point
/// `x`, `E[x0 | x]`. With `a = √ᾱ`, `b² = 1 − ᾱ`, blob k seen through the noise is a
/// round Gaussian around `a · centre_k` with variance `v = a² spread² + b²`:
/// - responsibility of blob k: `r_k ∝ exp(−|x − a centre_k|² / (2v))`, scaled to sum 1;
/// - best guess within blob k: `centre_k + (a spread² / v) (x − a centre_k)`;
/// - the answer: `Σ_k r_k · (best guess within blob k)`.
pub fn best_x0(x: &[f32], alpha_bar: f32, centres: &[Vec<f32>], spread: f32) -> Vec<f32> {
    let a = alpha_bar.sqrt();
    let v = a * a * spread * spread + (1.0 - alpha_bar);
    // log of the unnormalised responsibilities; subtract the largest before exp
    let logr: Vec<f32> = centres
        .iter()
        .map(|c| -(0..x.len()).map(|i| (x[i] - a * c[i]).powi(2)).sum::<f32>() / (2.0 * v))
        .collect();
    let top = logr.iter().cloned().fold(f32::MIN, f32::max);
    let r: Vec<f32> = logr.iter().map(|l| (l - top).exp()).collect();
    let total: f32 = r.iter().sum();
    let mut out = vec![0.0; x.len()];
    for (c, rk) in centres.iter().zip(&r) {
        for i in 0..x.len() {
            out[i] += rk / total * (c[i] + a * spread * spread / v * (x[i] - a * c[i]));
        }
    }
    out
}

/// C3. Deterministic sampling (DDIM): start at `x` (pure noise) with `alpha_bars[0]`, and go
/// through the signal levels `alpha_bars` (rising, ending at 1). At each step: estimate the
/// clean point `x̂0`, the noise `ε̂ = (x − √ᾱ x̂0) / √(1 − ᾱ)`, and move to the next level with
/// the same noise direction: `x = √ᾱ_next x̂0 + √(1 − ᾱ_next) ε̂`.
pub fn sample_ddim(x: &[f32], alpha_bars: &[f32], centres: &[Vec<f32>], spread: f32) -> Vec<f32> {
    let mut x = x.to_vec();
    for k in 0..alpha_bars.len() - 1 {
        let (ab, next) = (alpha_bars[k], alpha_bars[k + 1]);
        let x0 = best_x0(&x, ab, centres, spread);
        let eps: Vec<f32> = (0..x.len()).map(|i| (x[i] - ab.sqrt() * x0[i]) / (1.0 - ab).sqrt()).collect();
        x = forward(&x0, next, &eps);
    }
    x
}

/// C4. Stochastic sampling (DDPM): as C3, but each step draws the next point at random from
/// the Gaussian of the lecture's math slide. With the current level `ab`, the next `next`,
/// `α = ab / next` and `β = 1 − α`:
/// mean `(√next · β / (1 − ab)) x̂0 + (√α · (1 − next) / (1 − ab)) x`,
/// variance `β (1 − next) / (1 − ab)`.
pub fn sample_ddpm(x: &[f32], alpha_bars: &[f32], centres: &[Vec<f32>], spread: f32, rng: &mut Rng) -> Vec<f32> {
    let mut x = x.to_vec();
    for k in 0..alpha_bars.len() - 1 {
        let (ab, next) = (alpha_bars[k], alpha_bars[k + 1]);
        let x0 = best_x0(&x, ab, centres, spread);
        let alpha = ab / next;
        let beta = 1.0 - alpha;
        let c0 = next.sqrt() * beta / (1.0 - ab);
        let cx = alpha.sqrt() * (1.0 - next) / (1.0 - ab);
        let sd = (beta * (1.0 - next) / (1.0 - ab)).sqrt();
        x = (0..x.len()).map(|i| c0 * x0[i] + cx * x[i] + sd * rng.normal()).collect();
    }
    x
}

/// C5. Restoration of one value with a Gaussian prior `x ~ N(mu, s²)` from a measurement
/// `y = x + noise`, noise sd `sigma`: the posterior is Gaussian with
/// mean `mu + s²/(s² + sigma²) (y − mu)` and variance `s² sigma² / (s² + sigma²)`.
/// Returns `(mean, variance)`.
pub fn posterior(mu: f32, s: f32, sigma: f32, y: f32) -> (f32, f32) {
    let (s2, n2) = (s * s, sigma * sigma);
    (mu + s2 / (s2 + n2) * (y - mu), s2 * n2 / (s2 + n2))
}
