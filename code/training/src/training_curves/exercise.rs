// Coding exercises for "Reading training curves". Replace each `todo!()` with your code, then
// check it with `cargo test -p training training_curves::` and look at
// `cargo run -p training -- training-curves`.
#![allow(unused_variables, unused_imports)]

/// C1. A smoothed curve (exponential moving average): `s[0] = x[0]`, then
/// `s[t] = beta s[t - 1] + (1 - beta) x[t]`. Larger `beta` (closer to 1) smooths more.
///
/// Hint: push onto a new `Vec`, reading the value pushed just before.
pub fn smooth(x: &[f32], beta: f32) -> Vec<f32> {
    todo!("C1: smooth")
}

/// C2. Early stopping with patience: walk along the validation losses (one per epoch) and remember
/// the epoch with the lowest so far; stop at the first epoch that is `patience` epochs after the
/// best one. Returns (best epoch, epoch at which training stops); if it never stops early, the
/// stop epoch is the last one.
///
/// Hint: one loop from epoch 1; update the best first, then check `t - best >= patience`.
pub fn early_stopping(val: &[f32], patience: usize) -> (usize, usize) {
    todo!("C2: early_stopping")
}

/// C3. The loss of a network that has learnt only how common each class is: it gives every
/// example the probabilities f_k = counts[k] / total, so its mean cross-entropy is
/// `- sum over k of f_k ln f_k`. For K equally common classes this is ln K.
///
/// Hint: skip classes with count 0 (their term is 0).
pub fn chance_loss(counts: &[usize]) -> f32 {
    todo!("C3: chance_loss")
}

/// C4. Class weights against imbalance: `total / (K counts[k])`, so a rare class weighs more and
/// the weights, summed over all examples, still add up to the number of examples.
///
/// Hint: K is `counts.len()`.
pub fn class_weights(counts: &[usize]) -> Vec<f32> {
    todo!("C4: class_weights")
}

/// C5. The focal loss of one example: `-(1 - p)^gamma ln p`, with p the probability given to the
/// true class. Gamma 0 is the cross-entropy; larger gamma turns down the loss of the examples
/// that are already easy (p near 1).
///
/// Hint: `powf` and `ln`.
pub fn focal_loss(p: &[f32], true_class: usize, gamma: f32) -> f32 {
    todo!("C5: focal_loss")
}
