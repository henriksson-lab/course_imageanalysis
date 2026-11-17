// Model solution for the coding exercises of "SVMs for pixel classification".

/// C1a. The mean and the standard deviation of every feature (column) over the training
/// pixels, dividing by `n` (slide "Standardise the features"). Returns `(mean, sd)`.
pub fn fit_standardise(points: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
    let (n, d) = (points.len() as f32, points[0].len());
    let mut mean = vec![0.0; d];
    for p in points {
        for j in 0..d {
            mean[j] += p[j] / n;
        }
    }
    let mut sd = vec![0.0; d];
    for p in points {
        for j in 0..d {
            sd[j] += (p[j] - mean[j]).powi(2) / n;
        }
    }
    for s in sd.iter_mut() {
        *s = s.sqrt();
    }
    (mean, sd)
}

/// C1b. The z-scores of one feature vector: `z_j = (x_j − mean_j) / sd_j`, with the mean and
/// standard deviation of the training pixels.
pub fn standardise(x: &[f32], mean: &[f32], sd: &[f32]) -> Vec<f32> {
    (0..x.len()).map(|j| (x[j] - mean[j]) / sd[j]).collect()
}

/// C2. Balancing class weights (slide "Class imbalance"): class `c` gets `n / (k n_c)`, with
/// `n` training pixels, `k` classes and `n_c` pixels of class `c`. Classes are `0..k`.
pub fn class_weights(classes: &[usize], k: usize) -> Vec<f32> {
    let mut count = vec![0.0f32; k];
    for &c in classes {
        count[c] += 1.0;
    }
    let n = classes.len() as f32;
    count.iter().map(|&nc| n / (k as f32 * nc)).collect()
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
pub fn train_linear_svm(points: &[Vec<f32>], labels: &[f32], weights: &[f32], c: f32, steps: usize) -> Linear {
    let d = points[0].len();
    let objective = |m: &Linear| {
        let mut total = 0.5 * m.w.iter().map(|w| w * w).sum::<f32>();
        for i in 0..points.len() {
            total += c * weights[i] * (1.0 - labels[i] * m.f(&points[i])).max(0.0);
        }
        total
    };
    let mut model = Linear { w: vec![0.0; d], b: 0.0 };
    let mut best = (objective(&model), model.clone());
    for t in 1..=steps {
        let mut grad_w = model.w.clone();
        let mut grad_b = 0.0;
        for i in 0..points.len() {
            if labels[i] * model.f(&points[i]) < 1.0 {
                for j in 0..d {
                    grad_w[j] -= c * weights[i] * labels[i] * points[i][j];
                }
                grad_b -= c * weights[i] * labels[i];
            }
        }
        let length = (grad_w.iter().map(|g| g * g).sum::<f32>() + grad_b * grad_b).sqrt();
        if length == 0.0 {
            break;
        }
        let rate = 1.0 / ((t as f32).sqrt() * length);
        for j in 0..d {
            model.w[j] -= rate * grad_w[j];
        }
        model.b -= rate * grad_b;
        let value = objective(&model);
        if value < best.0 {
            best = (value, model.clone());
        }
    }
    best.1
}

/// C4a. One-vs-rest training (slide "More than two classes"): for every class `c` in `0..k`, a
/// linear SVM of class `c` (+1) against all others (−1). With `balanced`, every pixel gets
/// the weight of its class from `class_weights`, otherwise weight 1.
pub fn train_one_vs_rest(points: &[Vec<f32>], classes: &[usize], k: usize, c: f32, steps: usize, balanced: bool) -> Vec<Linear> {
    let class_w = class_weights(classes, k);
    let weights: Vec<f32> = classes.iter().map(|&cl| if balanced { class_w[cl] } else { 1.0 }).collect();
    (0..k)
        .map(|cl| {
            let labels: Vec<f32> = classes.iter().map(|&x| if x == cl { 1.0 } else { -1.0 }).collect();
            train_linear_svm(points, &labels, &weights, c, steps)
        })
        .collect()
}

/// C4b. One-vs-rest prediction: the class whose classifier gives the largest `f`.
pub fn predict_one_vs_rest(models: &[Linear], x: &[f32]) -> usize {
    let mut best = 0;
    for c in 1..models.len() {
        if models[c].f(x) > models[best].f(x) {
            best = c;
        }
    }
    best
}

/// C5. The cross-validation error (slide "(C, γ) by cross-validation"): pixel `i` belongs to
/// fold `folds[i]` (in `0..n_folds`). For every fold, train on all other folds and predict the
/// held-out one with `fit_predict(train points, train classes, held-out points)`. Returns the
/// fraction of all pixels that were predicted wrongly.
pub fn cross_validate(
    points: &[Vec<f32>],
    classes: &[usize],
    folds: &[usize],
    n_folds: usize,
    fit_predict: impl Fn(&[Vec<f32>], &[usize], &[Vec<f32>]) -> Vec<usize>,
) -> f32 {
    let mut wrong = 0;
    for fold in 0..n_folds {
        let (mut train, mut train_classes, mut held_out, mut truth) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..points.len() {
            if folds[i] == fold {
                held_out.push(points[i].clone());
                truth.push(classes[i]);
            } else {
                train.push(points[i].clone());
                train_classes.push(classes[i]);
            }
        }
        if held_out.is_empty() {
            continue;
        }
        let predicted = fit_predict(&train, &train_classes, &held_out);
        wrong += predicted.iter().zip(&truth).filter(|(p, t)| p != t).count();
    }
    wrong as f32 / points.len() as f32
}
