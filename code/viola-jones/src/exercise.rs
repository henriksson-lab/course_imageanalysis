// Coding exercises for "Boosted detectors & the cascade". Replace each `todo!()` with your code,
// then check it with `cargo test -p viola-jones` and look at `cargo run -p viola-jones`.
#![allow(unused_variables, unused_imports)]

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
///
/// Hint: `order.sort_by(|&i, &j| values[i].total_cmp(&values[j]))` sorts the indices by value.
/// Start with the threshold below all values (`values[order[0]] - 1.0`): every example is called
/// a face and the error is the weight of the negatives. Passing example `i` changes the error by
/// `labels[i] * weights[i]`. The opposite polarity has the error `total - error`.
pub fn best_stump(values: &[f32], labels: &[f64], weights: &[f64]) -> (f32, f64, f64) {
    todo!("C1: best_stump")
}

/// C2. AdaBoost: `features[f][i]` is the value of feature `f` on example `i`. Start with equal
/// weights; in every round choose the best stump over all features, give it the vote
/// `alpha = ½ ln((1 − ε)/ε)`, multiply the weights of the examples it gets wrong by `e^alpha`
/// and the others by `e^−alpha`, and divide by their sum (slide "The math: AdaBoost").
///
/// Hint: `best_stump` for every feature, keep the one with the smallest error; clamp the
/// error away from 0 before taking the logarithm. `stump_says` gives `h = ±1`; the new weight is
/// `w · exp(−alpha · label · h)`.
pub fn adaboost(features: &[Vec<f32>], labels: &[f64], rounds: usize) -> Vec<Stump> {
    todo!("C2: adaboost")
}

/// C3. The score of the strong classifier, `sum over the stumps of alpha · h(value)`, for one
/// example; `feature_value(f)` gives the value of feature `f` on it. Positive: face.
///
/// Hint: one line with `iter().map(...).sum()`, or a loop.
pub fn score(stumps: &[Stump], feature_value: impl Fn(usize) -> f32) -> f64 {
    todo!("C3: score")
}

/// C4. The rates of a cascade: overall detection rate `D`, false-positive rate `F`, and the
/// expected number of features a background window needs, `n1 + n2 f1 + n3 f1 f2 + ...`
/// (slide "The math: cascade rates"; the false-positive rates are the pass rates of
/// background windows).
///
/// Hint: keep a running product of the false-positive rates: the share of background windows
/// that reach the next stage.
pub fn cascade_rates(stages: &[Stage]) -> (f64, f64, f64) {
    todo!("C4: cascade_rates")
}
