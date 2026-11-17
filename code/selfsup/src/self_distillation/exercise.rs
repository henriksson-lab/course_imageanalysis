// Coding exercises for "Self-distillation and foundation models". Replace each `todo!()` with your code,
// then check it with `cargo test -p selfsup self_distillation::` and look at `cargo run -p selfsup -- self-distillation`.
#![allow(unused_variables, unused_imports)]

/// C1. The teacher's or student's probabilities: softmax of `(logits − centre) / tau`
/// (subtract the largest value before `exp`). `centre` may be all zeros.
///
/// Hint: as the softmax of the contrastive unit, with `(z − c) / tau` as the scores.
pub fn softmax_t(logits: &[f32], centre: &[f32], tau: f32) -> Vec<f32> {
    todo!("C1: softmax_t")
}

/// C2. The self-distillation loss of one pair: the cross-entropy `−Σ_k t_k ln s_k` between the
/// teacher's probabilities `t` and the student's `s = softmax(student_logits / tau_s)`, and its
/// gradient with respect to the student's logits, `(s − t) / tau_s`. The teacher gets no gradient.
///
/// Hint: `softmax_t(student_logits, &zeros, tau_s)` gives `s`.
pub fn distill_loss(t: &[f32], student_logits: &[f32], tau_s: f32) -> (f32, Vec<f32>) {
    todo!("C2: distill_loss")
}

/// C3. The moving average: every teacher weight moves a fraction `1 − m` towards the
/// student's: `teacher = m · teacher + (1 − m) · student`.
pub fn ema(teacher: &mut [f32], student: &[f32], m: f32) {
    todo!("C3: ema")
}

/// C4. The entropy `−Σ p ln p` of a probability vector (terms with p = 0 count 0).
pub fn entropy(p: &[f32]) -> f32 {
    todo!("C4: entropy")
}
