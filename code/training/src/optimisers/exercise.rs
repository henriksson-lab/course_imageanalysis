// Coding exercises for "Optimisers and learning rates". Replace each `todo!()` with your code,
// then check it with `cargo test -p training optimisers::` and look at `cargo run -p training -- optimisers`.
#![allow(unused_variables, unused_imports)]

/// C1. One step of plain gradient descent: `w -= eta * g` for every weight.
///
/// Hint: one loop over the indices of `w`.
pub fn gd_step(w: &mut [f32], g: &[f32], eta: f32) {
    todo!("C1: gd_step")
}

/// C2. One step of gradient descent with momentum: the velocity keeps a share `beta` of
/// itself and adds the gradient, `v = beta v + g`; then `w -= eta * v`.
///
/// Hint: update `v[i]` first, then use the new `v[i]` for `w[i]`.
pub fn momentum_step(w: &mut [f32], v: &mut [f32], g: &[f32], eta: f32, beta: f32) {
    todo!("C2: momentum_step")
}

/// The running averages of Adam, one pair per weight, and the step counter.
#[derive(Clone, Debug)]
pub struct AdamState {
    /// Running mean of the gradient.
    pub m: Vec<f32>,
    /// Running mean of the squared gradient.
    pub v: Vec<f32>,
    /// Number of steps taken so far.
    pub t: i32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    /// Divide m and v by (1 - beta^t) (the usual choice); `false` shows what goes wrong without.
    pub bias_correction: bool,
}

impl AdamState {
    /// Zero averages for `n` weights; beta1 = 0.9, beta2 = 0.999, eps = 1e-8, with bias correction.
    pub fn new(n: usize) -> AdamState {
        AdamState { m: vec![0.0; n], v: vec![0.0; n], t: 0, beta1: 0.9, beta2: 0.999, eps: 1e-8, bias_correction: true }
    }
}

/// C3. One step of Adam. Count the step (`t += 1`); for every weight update the running means
/// `m = beta1 m + (1 - beta1) g` and `v = beta2 v + (1 - beta2) g^2`; with bias correction
/// divide them by `1 - beta1^t` and `1 - beta2^t`; then `w -= eta * m / (sqrt(v) + eps)`.
///
/// Hint: `s.beta1.powi(s.t)` is beta1 to the power t. Store the running means (not the
/// corrected ones) back into `s.m` and `s.v`.
pub fn adam_step(w: &mut [f32], s: &mut AdamState, g: &[f32], eta: f32) {
    todo!("C3: adam_step")
}

/// C4. The learning rate of step `step` (counted from 0) of `total`: a linear warm-up over the
/// first `warmup` steps, `eta_max (step + 1) / warmup`, then a cosine decay from `eta_max` to 0,
/// `eta_max (1 + cos(pi (step - warmup) / (total - warmup))) / 2`.
///
/// Hint: `std::f32::consts::PI`; convert the counts with `as f32`.
pub fn learning_rate(step: usize, total: usize, warmup: usize, eta_max: f32) -> f32 {
    todo!("C4: learning_rate")
}
