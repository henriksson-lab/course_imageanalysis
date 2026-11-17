// Coding exercises for "SVMs for pixel classification". Replace each `todo!()` with your
// code, then check it with `cargo test -p day34 pixel_classify::` and look at `cargo run -p day34 -- pixel-classify`.
#![allow(unused_variables, unused_imports)]

/// C1a. The mean and the standard deviation of every feature (column) over the training
/// pixels, dividing by `n` (slide "Standardise the features"). Returns `(mean, sd)`.
///
/// Hint: two passes over the points: first the means, then the squared deviations.
pub fn fit_standardise(points: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
    todo!("C1a: fit_standardise")
}

/// C1b. The z-scores of one feature vector: `z_j = (x_j − mean_j) / sd_j`, with the mean and
/// standard deviation of the training pixels.
pub fn standardise(x: &[f32], mean: &[f32], sd: &[f32]) -> Vec<f32> {
    todo!("C1b: standardise")
}

/// C2. Balancing class weights (slide "Class imbalance"): class `c` gets `n / (k n_c)`, with
/// `n` training pixels, `k` classes and `n_c` pixels of class `c`. Classes are `0..k`.
pub fn class_weights(classes: &[usize], k: usize) -> Vec<f32> {
    todo!("C2: class_weights")
}

/// A linear classifier `f(x) = w · x + b`.
#[derive(Clone, Debug)]
pub struct Linear {
    pub w: Vec<f32>,
    pub b: f32,
}

impl Linear {
    /// The decision value `f(x) = w · x + b`. (Given.)
    pub fn f(&self, x: &[f32]) -> f32 {
        self.w.iter().zip(x).map(|(w, x)| w * x).sum::<f32>() + self.b
    }
}

/// C3. A linear SVM with a weight per point, trained as in the previous units by walking
/// downhill on `½‖w‖² + C Σ_i weight_i · max(0, 1 − y_i f(x_i))`. Labels `y` are ±1. Steps of
/// length `1/√t` against the gradient; keep the classifier with the smallest objective seen.
///
/// Hint: as `train` in the maxmargin unit, with `C` replaced by `C · weight_i` for point `i`.
/// Use `.clone()` to remember the best `Linear`.
pub fn train_linear_svm(points: &[Vec<f32>], labels: &[f32], weights: &[f32], c: f32, steps: usize) -> Linear {
    todo!("C3: train_linear_svm")
}

/// C4a. One-vs-rest training (slide "More than two classes"): for every class `c` in `0..k`, a
/// linear SVM of class `c` (+1) against all others (−1). With `balanced`, every pixel gets
/// the weight of its class from `class_weights`, otherwise weight 1.
///
/// Hint: build the ±1 labels for class `c` and call `train_linear_svm` once per class.
pub fn train_one_vs_rest(points: &[Vec<f32>], classes: &[usize], k: usize, c: f32, steps: usize, balanced: bool) -> Vec<Linear> {
    todo!("C4a: train_one_vs_rest")
}

/// C4b. One-vs-rest prediction: the class whose classifier gives the largest `f`.
pub fn predict_one_vs_rest(models: &[Linear], x: &[f32]) -> usize {
    todo!("C4b: predict_one_vs_rest")
}

/// C5. The cross-validation error (slide "(C, γ) by cross-validation"): pixel `i` belongs to
/// fold `folds[i]` (in `0..n_folds`). For every fold, train on all other folds and predict the
/// held-out one with `fit_predict(train points, train classes, held-out points)`. Returns the
/// fraction of all pixels that were predicted wrongly.
///
/// Hint: for each fold, collect the training points and the held-out points in separate
/// vectors (`.clone()` each point), call `fit_predict`, and count the wrong predictions.
pub fn cross_validate(
    points: &[Vec<f32>],
    classes: &[usize],
    folds: &[usize],
    n_folds: usize,
    fit_predict: impl Fn(&[Vec<f32>], &[usize], &[Vec<f32>]) -> Vec<usize>,
) -> f32 {
    todo!("C5: cross_validate")
}
