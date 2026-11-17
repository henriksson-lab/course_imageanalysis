// Model solution for "Reading training curves".

/// C1. A smoothed curve (exponential moving average): `s[0] = x[0]`, then
/// `s[t] = beta s[t - 1] + (1 - beta) x[t]`. Larger `beta` (closer to 1) smooths more.
pub fn smooth(x: &[f32], beta: f32) -> Vec<f32> {
    let mut s = Vec::with_capacity(x.len());
    for (t, &v) in x.iter().enumerate() {
        s.push(if t == 0 { v } else { beta * s[t - 1] + (1.0 - beta) * v });
    }
    s
}

/// C2. Early stopping with patience: walk along the validation losses (one per epoch) and remember
/// the epoch with the lowest so far; stop at the first epoch that is `patience` epochs after the
/// best one. Returns (best epoch, epoch at which training stops); if it never stops early, the
/// stop epoch is the last one.
pub fn early_stopping(val: &[f32], patience: usize) -> (usize, usize) {
    let mut best = 0;
    for t in 1..val.len() {
        if val[t] < val[best] {
            best = t;
        }
        if t - best >= patience {
            return (best, t);
        }
    }
    (best, val.len() - 1)
}

/// C3. The loss of a network that has learnt only how common each class is: it gives every
/// example the probabilities f_k = counts[k] / total, so its mean cross-entropy is
/// `- sum over k of f_k ln f_k`. For K equally common classes this is ln K.
pub fn chance_loss(counts: &[usize]) -> f32 {
    let total: usize = counts.iter().sum();
    counts
        .iter()
        .filter(|&&n| n > 0)
        .map(|&n| {
            let f = n as f32 / total as f32;
            -f * f.ln()
        })
        .sum()
}

/// C4. Class weights against imbalance: `total / (K counts[k])`, so a rare class weighs more and
/// the weights, summed over all examples, still add up to the number of examples.
pub fn class_weights(counts: &[usize]) -> Vec<f32> {
    let total: usize = counts.iter().sum();
    let k = counts.len() as f32;
    counts.iter().map(|&n| total as f32 / (k * n as f32)).collect()
}

/// C5. The focal loss of one example: `-(1 - p)^gamma ln p`, with p the probability given to the
/// true class. Gamma 0 is the cross-entropy; larger gamma turns down the loss of the examples
/// that are already easy (p near 1).
pub fn focal_loss(p: &[f32], true_class: usize, gamma: f32) -> f32 {
    let q = p[true_class];
    -(1.0 - q).powf(gamma) * q.ln()
}
