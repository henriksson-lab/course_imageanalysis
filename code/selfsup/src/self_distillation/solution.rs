// Model solution for the coding exercises of "Self-distillation and foundation models".

/// C1. The teacher's or student's probabilities: softmax of `(logits − centre) / tau`
/// (subtract the largest value before `exp`). `centre` may be all zeros.
pub fn softmax_t(logits: &[f32], centre: &[f32], tau: f32) -> Vec<f32> {
    let s: Vec<f32> = logits.iter().zip(centre).map(|(z, c)| (z - c) / tau).collect();
    let m = s.iter().cloned().fold(f32::MIN, f32::max);
    let e: Vec<f32> = s.iter().map(|v| (v - m).exp()).collect();
    let total: f32 = e.iter().sum();
    e.iter().map(|v| v / total).collect()
}

/// C2. The self-distillation loss of one pair: the cross-entropy `−Σ_k t_k ln s_k` between the
/// teacher's probabilities `t` and the student's `s = softmax(student_logits / tau_s)`, and its
/// gradient with respect to the student's logits, `(s − t) / tau_s`. The teacher gets no gradient.
pub fn distill_loss(t: &[f32], student_logits: &[f32], tau_s: f32) -> (f32, Vec<f32>) {
    let zero = vec![0.0; student_logits.len()];
    let s = softmax_t(student_logits, &zero, tau_s);
    let loss = -t.iter().zip(&s).map(|(tk, sk)| tk * sk.max(1e-30).ln()).sum::<f32>();
    let grad = s.iter().zip(t).map(|(sk, tk)| (sk - tk) / tau_s).collect();
    (loss, grad)
}

/// C3. The moving average: every teacher weight moves a fraction `1 − m` towards the
/// student's: `teacher = m · teacher + (1 − m) · student`.
pub fn ema(teacher: &mut [f32], student: &[f32], m: f32) {
    for (t, s) in teacher.iter_mut().zip(student) {
        *t = m * *t + (1.0 - m) * s;
    }
}

/// C4. The entropy `−Σ p ln p` of a probability vector (terms with p = 0 count 0).
pub fn entropy(p: &[f32]) -> f32 {
    -p.iter().filter(|&&v| v > 0.0).map(|v| v * v.ln()).sum::<f32>()
}
