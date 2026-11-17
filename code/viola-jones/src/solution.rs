// Model solution for the coding exercises of "Boosted detectors & the cascade".

/// A decision stump chosen by boosting: one feature, one threshold, one direction, and its
/// vote. `polarity` is `+1.0` for "face if the value is above the threshold" and `-1.0` for
/// "face if the value is below".
#[derive(Clone, Copy, Debug)]
pub struct Stump {
    pub feature: usize,
    pub threshold: f32,
    pub polarity: f64,
    pub error: f64,
    pub alpha: f64,
}

/// What a stump says about a value (given): `+1.0` (face) or `-1.0` (background).
pub fn stump_says(threshold: f32, polarity: f64, value: f32) -> f64 {
    if value > threshold { polarity } else { -polarity }
}

/// A stage of a cascade: its detection rate `d`, false-positive rate `f` (on the windows that
/// reach it) and its number of features `n`.
#[derive(Clone, Copy, Debug)]
pub struct Stage {
    pub d: f64,
    pub f: f64,
    pub n: usize,
}

/// C1. The best stump for one feature: `values[i]` is the feature's value on example `i`,
/// `labels[i]` is `+1.0` (face) or `-1.0`, `weights[i]` its weight. Returns `(threshold,
/// polarity, weighted error)`. Sort the examples by value once; then move the threshold past
/// one example at a time and update the error with a running sum (slide "The math: the best
/// threshold in one pass"). Thresholds lie halfway between neighbouring values; of equally good
/// stumps, keep the first one found.
pub fn best_stump(values: &[f32], labels: &[f64], weights: &[f64]) -> (f32, f64, f64) {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| values[i].total_cmp(&values[j]));
    let total: f64 = weights.iter().sum();
    // threshold below all values: every example is called a face, the negatives are wrong
    let mut error: f64 = (0..n).filter(|&i| labels[i] < 0.0).map(|i| weights[i]).sum();
    let mut best = (values[order[0]] - 1.0, 1.0, error);
    if total - error < best.2 {
        best = (best.0, -1.0, total - error);
    }
    for k in 0..n {
        let i = order[k];
        // example i moves below the threshold: it is now called background
        error += labels[i] * weights[i];
        if k + 1 < n && values[order[k + 1]] == values[i] {
            continue; // no threshold fits between equal values
        }
        let threshold = if k + 1 < n { 0.5 * (values[i] + values[order[k + 1]]) } else { values[i] + 1.0 };
        if error < best.2 {
            best = (threshold, 1.0, error);
        }
        if total - error < best.2 {
            best = (threshold, -1.0, total - error);
        }
    }
    best
}

/// C2. AdaBoost: `features[f][i]` is the value of feature `f` on example `i`. Start with equal
/// weights; in every round choose the best stump over all features, give it the vote
/// `alpha = ½ ln((1 − ε)/ε)`, multiply the weights of the examples it gets wrong by `e^alpha`
/// and the others by `e^−alpha`, and divide by their sum (slide "The math: AdaBoost").
pub fn adaboost(features: &[Vec<f32>], labels: &[f64], rounds: usize) -> Vec<Stump> {
    let n = labels.len();
    let mut weights = vec![1.0 / n as f64; n];
    let mut stumps = Vec::new();
    for _ in 0..rounds {
        let mut best: Option<Stump> = None;
        for (f, values) in features.iter().enumerate() {
            let (threshold, polarity, error) = best_stump(values, labels, &weights);
            if best.is_none_or(|b| error < b.error) {
                best = Some(Stump { feature: f, threshold, polarity, error, alpha: 0.0 });
            }
        }
        let mut s = best.expect("no features");
        let eps = s.error.clamp(1e-10, 1.0 - 1e-10);
        s.alpha = 0.5 * ((1.0 - eps) / eps).ln();
        for i in 0..n {
            let h = stump_says(s.threshold, s.polarity, features[s.feature][i]);
            weights[i] *= (-s.alpha * labels[i] * h).exp();
        }
        let sum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= sum;
        }
        stumps.push(s);
    }
    stumps
}

/// C3. The score of the strong classifier, `sum over the stumps of alpha · h(value)`, for one
/// example; `feature_value(f)` gives the value of feature `f` on it. Positive: face.
pub fn score(stumps: &[Stump], feature_value: impl Fn(usize) -> f32) -> f64 {
    stumps.iter().map(|s| s.alpha * stump_says(s.threshold, s.polarity, feature_value(s.feature))).sum()
}

/// C4. The rates of a cascade: overall detection rate `D`, false-positive rate `F`, and the
/// expected number of features a background window needs, `n1 + n2 f1 + n3 f1 f2 + ...`
/// (slide "The math: cascade rates"; the false-positive rates are the pass rates of
/// background windows).
pub fn cascade_rates(stages: &[Stage]) -> (f64, f64, f64) {
    let (mut d, mut f, mut features) = (1.0, 1.0, 0.0);
    for s in stages {
        features += s.n as f64 * f; // only the windows that got this far are evaluated
        d *= s.d;
        f *= s.f;
    }
    (d, f, features)
}
