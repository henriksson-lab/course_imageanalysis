// Coding exercises for "Generative adversarial networks". Replace each `todo!()` with your code,
// then check it with `cargo test -p gans gan::` and look at `cargo run -p gans -- gan`.
#![allow(unused_variables, unused_imports, unused_mut)]

pub use crate::common::{sigmoid, GenLoss, LineGan, Mixture, Settings};
use crate::common::{gauss, log_sigmoid};
use imgcourse::Rng;

/// C1. The discriminator D(x) = σ(c0 + c1 x + c2 x²): the probability that x is real.
pub fn discriminator(c: &[f64; 3], x: f64) -> f64 {
    todo!("C1: discriminator")
}

/// C2. The best discriminator for data density `p_data` and generator density `p_gen` at a point:
/// D*(x) = p_data / (p_data + p_gen) (one half where both are zero).
pub fn optimal_discriminator(p_data: f64, p_gen: f64) -> f64 {
    todo!("C2: optimal_discriminator")
}

/// C3. The discriminator's loss: the mean of −log D(x) over the real points plus the mean of
/// −log(1 − D(x)) over the generated points.
pub fn loss_d(c: &[f64; 3], real: &[f64], fake: &[f64]) -> f64 {
    todo!("C3: loss_d")
}

/// C3. The gradient of `loss_d` with respect to (c0, c1, c2).
pub fn grad_d(c: &[f64; 3], real: &[f64], fake: &[f64]) -> [f64; 3] {
    todo!("C3: grad_d")
}

/// C4. The gradient of the generator's loss (mean over the noise values `zs`) with respect to
/// (a, b), for the generator x = a z + b and the discriminator with weights `gan.c`.
pub fn grad_g(gan: &LineGan, zs: &[f64], loss: GenLoss) -> (f64, f64) {
    todo!("C4: grad_g")
}

/// C5. One training step: `k_d` discriminator steps (each with a fresh batch of real points
/// from `data` and generated points), then one generator step with fresh noise. Plain gradient
/// descent with the learning rates of `s` (`s.optimal` is not used here).
pub fn train_step(gan: &mut LineGan, data: &Mixture, s: &Settings, rng: &mut Rng) {
    todo!("C5: train_step")
}
