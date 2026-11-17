// Coding exercises for "Fine-tuning strategies". Replace each `todo!()` with your code,
// then check it with `cargo test -p finetuning finetune_strategies::` and look at
// `cargo run -p finetuning -- finetune-strategies`.
#![allow(unused_variables, unused_imports)]

pub use crate::common::*;

/// C1. The penalty λ Σ (w − w₀)² that keeps the weights near the pretrained ones ("L2-SP"):
/// add its gradient 2λ (w − w₀) to the gradient `g` of every weight of the backbone: all layers
/// but the last (the new head has no pretrained weights to stay near), and not the biases.
///
/// Hint: three nested loops over layers `l` (up to `net.len() - 1`, not included), outputs `o`
/// and inputs `i` of `net[l].w[o][i]`.
pub fn add_pull_to_start(g: &mut Net, net: &Net, start: &Net, lambda: f32) {
    todo!("C1: add_pull_to_start")
}

/// C2. Numbers to store for a change of an m × n weight matrix: the full change (m·n), and a
/// low-rank change B·A with B m × r and A r × n.
pub fn param_counts(m: usize, n: usize, r: usize) -> (usize, usize) {
    todo!("C2: param_counts")
}

/// The matrix product of an a × b and a b × c matrix (lists of rows).
pub fn matmul(x: &[Vec<f32>], y: &[Vec<f32>]) -> Vec<Vec<f32>> {
    x.iter().map(|row| (0..y[0].len()).map(|j| row.iter().zip(y).map(|(v, yr)| v * yr[j]).sum()).collect()).collect()
}

/// C3. Fit a rank-r change B·A to a target change Δ (m × n) by gradient descent on
/// ‖B·A − Δ‖² (sum of squares). Start as LoRA does: A small and random (0.1 · normal), B = 0.
/// Gradients: with E = B·A − Δ, ∂/∂B = 2 E Aᵀ and ∂/∂A = 2 Bᵀ E. Returns (B, A).
///
/// Hint: `imgcourse::Rng::new(seed)` and `rng.normal()` for A (r rows of n numbers), then in
/// every step: `matmul(&b, &a)`, E, both gradients (from the same E), then both updates.
pub fn fit_low_rank(delta: &[Vec<f32>], r: usize, steps: usize, lr: f32, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    todo!("C3: fit_low_rank")
}

/// C4. The relative error ‖B·A − Δ‖ / ‖Δ‖ (square roots of sums of squares).
pub fn rel_error(delta: &[Vec<f32>], b: &[Vec<f32>], a: &[Vec<f32>]) -> f32 {
    todo!("C4: rel_error")
}
