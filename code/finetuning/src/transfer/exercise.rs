// Coding exercises for "Transfer learning". Replace each `todo!()` with your code,
// then check it with `cargo test -p finetuning transfer::` and look at `cargo run -p finetuning -- transfer`.
#![allow(unused_variables, unused_imports)]

pub use crate::common::*;

/// C1. The features of a patch after the first `layers` layers of a network (each followed by
/// its ReLU): the frozen backbone's description of the patch.
///
/// Hint: start with `a = x.to_vec()`; for each of the first `layers` layers, the new `a[o]` is
/// `max(0, b[o] + sum over i of w[o][i] * a[i])`.
pub fn features(net: &Net, x: &[f32], layers: usize) -> Vec<f32> {
    todo!("C1: features")
}

/// C2. A linear probe: softmax regression on fixed features, `steps` steps of full-batch
/// gradient descent with learning rate `lr`, starting from all weights 0. The gradient of the
/// mean cross-entropy for one example is `(p − y) fᵀ` for the weights and `p − y` for the biases.
///
/// Hint: `Layer { w: vec![vec![0.0; d]; k], b: vec![0.0; k] }` is the start; in every step add
/// up the gradients over all examples (divided by their number) in `probe.zeros_like()`, using
/// `softmax(&scores(&probe, f))` for p, then subtract `lr` times them.
pub fn train_probe(feats: &[Vec<f32>], ys: &[usize], k: usize, steps: usize, lr: f32) -> Layer {
    todo!("C2: train_probe")
}

/// The class scores of a probe: `W f + b`.
pub fn scores(probe: &Layer, f: &[f32]) -> Vec<f32> {
    probe.w.iter().zip(&probe.b).map(|(row, b)| b + row.iter().zip(f).map(|(w, v)| w * v).sum::<f32>()).collect()
}

/// C3. The fraction of examples whose highest probe score is the true class.
///
/// Hint: `argmax(&scores(probe, f))` is the predicted class.
pub fn probe_accuracy(probe: &Layer, feats: &[Vec<f32>], ys: &[usize]) -> f32 {
    todo!("C3: probe_accuracy")
}

/// C4. How many weights and biases are trained when the first `frozen` layers are frozen.
///
/// Hint: `layer.count()` counts the weights and biases of one layer.
pub fn trained_weights(net: &Net, frozen: usize) -> usize {
    todo!("C4: trained_weights")
}
