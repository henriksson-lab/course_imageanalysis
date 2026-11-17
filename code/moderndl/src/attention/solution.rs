// Model solution for the coding exercises of "Attention, transformers and pretrained models".

use imgcourse::Image;

/// R1. Softmax of `scores`, and the weighted average of `values` with these weights.
/// Subtract the largest score before `exp`, so that large scores do not overflow
/// (this does not change the weights).
pub fn weights_and_average(scores: &[f32], values: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
    let top = scores.iter().cloned().fold(f32::MIN, f32::max);
    let e: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
    let total: f32 = e.iter().sum();
    let weights: Vec<f32> = e.iter().map(|v| v / total).collect();
    let mut out = vec![0.0; values[0].len()];
    for (w, v) in weights.iter().zip(values) {
        for d in 0..out.len() {
            out[d] += w * v[d];
        }
    }
    (weights, out)
}

/// R2. Attention for one query: the score of token `j` is `query · keys[j] / temperature`;
/// the weights are the softmax of the scores (positive, sum 1); the output is the weighted
/// average of the values. Returns `(weights, output)`.
pub fn attend(query: &[f32], keys: &[Vec<f32>], values: &[Vec<f32>], temperature: f32) -> (Vec<f32>, Vec<f32>) {
    let mut scores = Vec::new();
    for k in keys {
        let mut s = 0.0;
        for d in 0..query.len() {
            s += query[d] * k[d];
        }
        scores.push(s / temperature);
    }
    weights_and_average(&scores, values)
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
pub fn patch_attention(img: &Image<f32>, p: usize, content: f32, position: f32, temperature: f32) -> Image<f32> {
    let patches = to_patches(img, p);
    let per_row = img.width() / p;
    let mut out = Vec::new();
    for i in 0..patches.len() {
        let (xi, yi) = ((i % per_row) as f32, (i / per_row) as f32);
        let mut scores = Vec::new();
        for j in 0..patches.len() {
            let (xj, yj) = ((j % per_row) as f32, (j / per_row) as f32);
            let mut diff = 0.0;
            for k in 0..p * p {
                diff += (patches[i][k] - patches[j][k]).powi(2) / (p * p) as f32;
            }
            let grid = (xi - xj).powi(2) + (yi - yj).powi(2);
            scores.push((-content * diff - position * grid) / temperature);
        }
        out.push(weights_and_average(&scores, &patches).1);
    }
    from_patches(&out, p, img.width())
}

/// R4. Expected calibration error: put the predictions in `bins` equal bins of confidence
/// between 0.5 and 1 (a confidence of exactly 1 goes in the last bin); per bin compare
/// the accuracy with the mean confidence; `ECE = Σ_b (n_b / n) |acc_b − conf_b|`.
pub fn ece(confidence: &[f32], correct: &[bool], bins: usize) -> f32 {
    let n = confidence.len() as f32;
    let mut count = vec![0.0; bins];
    let mut hits = vec![0.0; bins];
    let mut conf_sum = vec![0.0; bins];
    for (c, ok) in confidence.iter().zip(correct) {
        let b = (((c - 0.5) / 0.5 * bins as f32) as usize).min(bins - 1);
        count[b] += 1.0;
        conf_sum[b] += c;
        if *ok {
            hits[b] += 1.0;
        }
    }
    let mut total = 0.0;
    for b in 0..bins {
        if count[b] > 0.0 {
            total += count[b] / n * (hits[b] / count[b] - conf_sum[b] / count[b]).abs();
        }
    }
    total
}

/// The logistic function `1 / (1 + e^(−z))`: the probability of class 1 for the logit `z`.
pub fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// R5. Temperature scaling for two classes: try every `T` in `candidates` and return the
/// one with the smallest negative log-likelihood `−(1/n) Σ log p_i(true class)`, where
/// `p = sigmoid(logit / T)` is the probability of class 1 (`labels[i]` true means class 1).
pub fn fit_temperature(logits: &[f32], labels: &[bool], candidates: &[f32]) -> f32 {
    let mut best = (candidates[0], f32::INFINITY);
    for &t in candidates {
        let mut nll = 0.0;
        for (z, &y) in logits.iter().zip(labels) {
            let p = sigmoid(z / t);
            let p_true = if y { p } else { 1.0 - p };
            nll -= p_true.max(1e-12).ln() / logits.len() as f32;
        }
        if nll < best.1 {
            best = (t, nll);
        }
    }
    best.0
}
