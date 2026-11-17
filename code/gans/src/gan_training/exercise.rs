// Coding exercises for "Training GANs and what goes wrong". Replace each `todo!()` with your code,
// then check it with `cargo test -p gans gan_training::` and look at `cargo run -p gans -- gan-training`.
#![allow(unused_variables, unused_imports, unused_mut)]

pub use crate::common::{GenLoss, LineGan, Mixture, Settings};

/// C1. Jensen–Shannon divergence between two densities given on the same grid of spacing `dx`:
/// JS = ½ Σ p log(2p/(p + q)) dx + ½ Σ q log(2q/(p + q)) dx, with 0 log 0 = 0. At most ln 2.
pub fn js_divergence(p: &[f64], q: &[f64], dx: f64) -> f64 {
    todo!("C1: js_divergence")
}

/// C2. Wasserstein-1 distance ("earth mover's distance") between two equally large samples in
/// 1D: sort both, then the mean of |a_i − b_i| over matching ranks.
pub fn wasserstein1(a: &[f64], b: &[f64]) -> f64 {
    todo!("C2: wasserstein1")
}

/// C3. For every point of `xs`: the distance to its k-th nearest other point of `xs` (1D).
pub fn knn_radii(xs: &[f64], k: usize) -> Vec<f64> {
    todo!("C3: knn_radii")
}

/// C3. Precision and recall of generated points `fake` against real points `real` (1D):
/// precision = share of fake points within the k-NN radius of some real point (realism);
/// recall = share of real points within the k-NN radius of some fake point (coverage).
pub fn precision_recall(real: &[f64], fake: &[f64], k: usize) -> (f64, f64) {
    todo!("C3: precision_recall")
}
