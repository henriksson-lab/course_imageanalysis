// Model solution for the coding exercises of "The variational autoencoder".
//
// The one-dimensional VAE of the lecture: one grey value x, a latent z, the decoder x̂ = w z
// with pixel noise sigma_x, and the encoder's cloud N(mu, s²).

/// C1a. A draw from the cloud by the reparameterisation: `z = mu + s · eps`.
pub fn reparameterise(mu: f32, s: f32, eps: f32) -> f32 {
    mu + s * eps
}

/// C1b. The KL divergence of the cloud `N(mu, s²)` (one entry per latent dimension) to the
/// prior `N(0, I)`: `½ (mu² + s² − 1 − ln s²)` per dimension, added over the dimensions.
pub fn kl_to_prior(mu: &[f32], s: &[f32]) -> f32 {
    let mut kl = 0.0;
    for d in 0..mu.len() {
        kl += 0.5 * (mu[d] * mu[d] + s[d] * s[d] - 1.0 - (s[d] * s[d]).ln());
    }
    kl
}

/// C2a. The reconstruction term, exactly: the average over the cloud of the squared error
/// divided by `2 sigma_x²`, `R = ((x − w mu)² + w² s²) / (2 sigma_x²)`.
pub fn reconstruction_exact(x: f32, w: f32, sigma_x: f32, mu: f32, s: f32) -> f32 {
    ((x - w * mu).powi(2) + w * w * s * s) / (2.0 * sigma_x * sigma_x)
}

/// C2b. The same term estimated as training does: draw `z = mu + s eps` for every given
/// `eps`, decode, and average `(x − w z)² / (2 sigma_x²)` over the draws.
pub fn reconstruction_from_draws(x: f32, w: f32, sigma_x: f32, mu: f32, s: f32, eps: &[f32]) -> f32 {
    let mut sum = 0.0;
    for &e in eps {
        let z = reparameterise(mu, s, e);
        sum += (x - w * z).powi(2) / (2.0 * sigma_x * sigma_x);
    }
    sum / eps.len() as f32
}

/// C3. Find the best cloud for the image `x` as a VAE is trained: gradient descent on
/// `R + beta · KL`, with `R` estimated from `draws` random draws per step (the
/// reparameterisation makes it a formula in `mu` and `s`, so it has a gradient).
/// The spread is stored as `log s`, so that it stays positive. Starts at the prior,
/// `mu = 0, s = 1`. Returns the path `(mu, s)` after every step, the start included.
#[allow(clippy::too_many_arguments)]
pub fn fit_cloud(x: f32, w: f32, sigma_x: f32, beta: f32, steps: usize, draws: usize, rate: f32, seed: u64) -> Vec<(f32, f32)> {
    let mut rng = imgcourse::Rng::new(seed);
    let (mut mu, mut log_s) = (0.0f32, 0.0f32);
    let mut path = vec![(mu, 1.0)];
    for _ in 0..steps {
        let s = log_s.exp();
        // the reconstruction term's gradient, averaged over the draws:
        // d/dmu (x − w z)²/(2σ²) = −w (x − w z)/σ², and dz/ds = eps
        let (mut g_mu, mut g_s) = (0.0, 0.0);
        for _ in 0..draws {
            let eps = rng.normal();
            let z = reparameterise(mu, s, eps);
            let d = -w * (x - w * z) / (sigma_x * sigma_x);
            g_mu += d / draws as f32;
            g_s += d * eps / draws as f32;
        }
        // plus beta times the gradient of the KL: d/dmu = mu, d/ds = s − 1/s
        g_mu += beta * mu;
        g_s += beta * (s - 1.0 / s);
        mu -= rate * g_mu;
        log_s -= rate * g_s * s; // d/d(log s) = s · d/ds
        path.push((mu, log_s.exp()));
    }
    path
}
