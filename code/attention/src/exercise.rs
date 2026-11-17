// Coding exercises for "Attention, transformers and pretrained models". Replace each `todo!()` with your code,
// then check it with `cargo test -p attention` and look at `cargo run -p attention`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// R1. Softmax of `scores`, and the weighted average of `values` with these weights.
/// Subtract the largest score before `exp`, so that large scores do not overflow
/// (this does not change the weights).
///
/// Hint: `scores.iter().cloned().fold(f32::MIN, f32::max)` is the largest score.
pub fn weights_and_average(scores: &[f32], values: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
    todo!("R1: weights_and_average")
}

/// R2. Attention for one query: the score of token `j` is `query · keys[j] / temperature`;
/// the weights are the softmax of the scores (positive, sum 1); the output is the weighted
/// average of the values. Returns `(weights, output)`.
///
/// Hint: compute the scores, then use `weights_and_average`.
pub fn attend(query: &[f32], keys: &[Vec<f32>], values: &[Vec<f32>], temperature: f32) -> (Vec<f32>, Vec<f32>) {
    todo!("R2: attend")
}

/// Cut an image into `p × p` patches, row by row of patches: each patch flattened into a
/// vector of `p²` grey values. (Width and height must be multiples of `p`.)
pub fn to_patches(img: &Image<f32>, p: usize) -> Vec<Vec<f32>> {
    let mut out = Vec::new();
    for y0 in (0..img.height()).step_by(p) {
        for x0 in (0..img.width()).step_by(p) {
            out.push(img.crop(x0, y0, p, p).pixels().to_vec());
        }
    }
    out
}

/// Put patches back together into an image `width` pixels wide (the inverse of `to_patches`).
pub fn from_patches(patches: &[Vec<f32>], p: usize, width: usize) -> Image<f32> {
    let per_row = width / p;
    let height = patches.len() / per_row * p;
    Image::from_fn(width, height, |x, y| patches[(y / p) * per_row + x / p][(y % p) * p + x % p])
}

/// R3. One attention layer on the patches of an image, with hand-set scores as in the demo:
/// the score of patch `j` for patch `i` is
/// `−content · (mean squared difference of their pixels) − position · (grid distance)²`,
/// divided by `temperature`; the output patch `i` is the weighted average of all patches.
///
/// Hint: `to_patches` cuts the image, `from_patches` puts the output patches together. Patch
/// `i` sits at grid position `(i % per_row, i / per_row)` with `per_row = img.width() / p`.
pub fn patch_attention(img: &Image<f32>, p: usize, content: f32, position: f32, temperature: f32) -> Image<f32> {
    todo!("R3: patch_attention")
}

/// R4. Expected calibration error: put the predictions in `bins` equal bins of confidence
/// between 0.5 and 1 (a confidence of exactly 1 goes in the last bin); per bin compare
/// the accuracy with the mean confidence; `ECE = Σ_b (n_b / n) |acc_b − conf_b|`.
///
/// Hint: the bin of a confidence `c` is `((c − 0.5) / 0.5 · bins)` rounded down, at most `bins
/// − 1`. Keep per bin the count, the number correct and the sum of the confidences.
pub fn ece(confidence: &[f32], correct: &[bool], bins: usize) -> f32 {
    todo!("R4: ece")
}

/// The logistic function `1 / (1 + e^(−z))`: the probability of class 1 for the logit `z`.
pub fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// R5. Temperature scaling for two classes: try every `T` in `candidates` and return the
/// one with the smallest negative log-likelihood `−(1/n) Σ log p_i(true class)`, where
/// `p = sigmoid(logit / T)` is the probability of class 1 (`labels[i]` true means class 1).
///
/// Hint: for every candidate `t`, add up `−ln p` of the true class over all examples; `p =
/// sigmoid(z / t)` for class 1 and `1 − p` for class 0.
pub fn fit_temperature(logits: &[f32], labels: &[bool], candidates: &[f32]) -> f32 {
    todo!("R5: fit_temperature")
}
