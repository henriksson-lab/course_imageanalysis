// Coding exercises for "The variational autoencoder". Replace each `todo!()` with your code,
// then check it with `cargo test -p day38 vae::` and look at `cargo run -p day38 -- vae`.
#![allow(unused_variables, unused_imports)]

// The one-dimensional VAE of the lecture: one grey value x, a latent z, the decoder x̂ = w z
// with pixel noise sigma_x, and the encoder's cloud N(mu, s²).

/// C1a. A draw from the cloud by the reparameterisation: `z = mu + s · eps`.
pub fn reparameterise(mu: f32, s: f32, eps: f32) -> f32 {
    todo!("C1a: reparameterise")
}

/// C1b. The KL divergence of the cloud `N(mu, s²)` (one entry per latent dimension) to the
/// prior `N(0, I)`: `½ (mu² + s² − 1 − ln s²)` per dimension, added over the dimensions.
///
/// Hint: `(s[d] * s[d]).ln()` is ln s².
pub fn kl_to_prior(mu: &[f32], s: &[f32]) -> f32 {
    todo!("C1b: kl_to_prior")
}

/// C2a. The reconstruction term, exactly: the average over the cloud of the squared error
/// divided by `2 sigma_x²`, `R = ((x − w mu)² + w² s²) / (2 sigma_x²)`.
pub fn reconstruction_exact(x: f32, w: f32, sigma_x: f32, mu: f32, s: f32) -> f32 {
    todo!("C2a: reconstruction_exact")
}

/// C2b. The same term estimated as training does: draw `z = mu + s eps` for every given
/// `eps`, decode, and average `(x − w z)² / (2 sigma_x²)` over the draws.
///
/// Hint: use `reparameterise` for every `eps`, then average.
pub fn reconstruction_from_draws(x: f32, w: f32, sigma_x: f32, mu: f32, s: f32, eps: &[f32]) -> f32 {
    todo!("C2b: reconstruction_from_draws")
}

/// C3. Find the best cloud for the image `x` as a VAE is trained: gradient descent on
/// `R + beta · KL`, with `R` estimated from `draws` random draws per step (the
/// reparameterisation makes it a formula in `mu` and `s`, so it has a gradient).
/// The spread is stored as `log s`, so that it stays positive. Starts at the prior,
/// `mu = 0, s = 1`. Returns the path `(mu, s)` after every step, the start included.
#[allow(clippy::too_many_arguments)]
///
/// Hint: keep `mu` and `log_s`; in every step set `s = log_s.exp()`, draw `eps = rng.normal()`
/// `draws` times and average the gradient of the reconstruction term: for one draw it is `d =
/// −w (x − w z) / sigma_x²` for mu and `d · eps` for s. Add the KL's gradient, `beta · mu` for
/// mu and `beta · (s − 1/s)` for s. Step: `mu −= rate · g_mu`, `log_s −= rate · g_s · s`.
pub fn fit_cloud(x: f32, w: f32, sigma_x: f32, beta: f32, steps: usize, draws: usize, rate: f32, seed: u64) -> Vec<(f32, f32)> {
    todo!("C3: fit_cloud")
}
