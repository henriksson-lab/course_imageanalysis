// Model solution for "Optimisers and learning rates".

/// C1. One step of plain gradient descent: `w -= eta * g` for every weight.
pub fn gd_step(w: &mut [f32], g: &[f32], eta: f32) {
    for i in 0..w.len() {
        w[i] -= eta * g[i];
    }
}

/// C2. One step of gradient descent with momentum: the velocity keeps a share `beta` of
/// itself and adds the gradient, `v = beta v + g`; then `w -= eta * v`.
pub fn momentum_step(w: &mut [f32], v: &mut [f32], g: &[f32], eta: f32, beta: f32) {
    for i in 0..w.len() {
        v[i] = beta * v[i] + g[i];
        w[i] -= eta * v[i];
    }
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
pub fn adam_step(w: &mut [f32], s: &mut AdamState, g: &[f32], eta: f32) {
    s.t += 1;
    let (c1, c2) = if s.bias_correction { (1.0 - s.beta1.powi(s.t), 1.0 - s.beta2.powi(s.t)) } else { (1.0, 1.0) };
    for i in 0..w.len() {
        s.m[i] = s.beta1 * s.m[i] + (1.0 - s.beta1) * g[i];
        s.v[i] = s.beta2 * s.v[i] + (1.0 - s.beta2) * g[i] * g[i];
        let m_hat = s.m[i] / c1;
        let v_hat = s.v[i] / c2;
        w[i] -= eta * m_hat / (v_hat.sqrt() + s.eps);
    }
}

/// C4. The learning rate of step `step` (counted from 0) of `total`: a linear warm-up over the
/// first `warmup` steps, `eta_max (step + 1) / warmup`, then a cosine decay from `eta_max` to 0,
/// `eta_max (1 + cos(pi (step - warmup) / (total - warmup))) / 2`.
pub fn learning_rate(step: usize, total: usize, warmup: usize, eta_max: f32) -> f32 {
    if step < warmup {
        return eta_max * (step + 1) as f32 / warmup as f32;
    }
    let progress = (step - warmup) as f32 / (total - warmup).max(1) as f32;
    0.5 * eta_max * (1.0 + (std::f32::consts::PI * progress).cos())
}
