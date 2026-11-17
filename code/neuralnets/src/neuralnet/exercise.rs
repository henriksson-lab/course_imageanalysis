// Coding exercises for "Neural networks: matrix, bend, matrix". Replace each `todo!()` with your code,
// then check it with `cargo test -p neuralnets neuralnet::` and look at `cargo run -p neuralnets -- neuralnet`.
#![allow(unused_variables, unused_imports)]

/// C1a. One neuron before its bend: the weighted sum `w1 x1 + w2 x2 + ... + b`.
/// (Apply `sigmoid` or `relu` to the result to get the neuron's output.)
///
/// Hint: start with `b`, then add `w[i] * x[i]` for every `i`.
pub fn weighted_sum(w: &[f32], b: f32, x: &[f32]) -> f32 {
    todo!("C1a: weighted_sum")
}

/// C1b. The sigmoid `1 / (1 + e^(-s))`: a number between 0 and 1.
///
/// Hint: `(-s).exp()` is e to the power -s.
pub fn sigmoid(s: f32) -> f32 {
    todo!("C1b: sigmoid")
}

/// C1c. The ReLU `max(0, s)`: negatives become 0.
pub fn relu(s: f32) -> f32 {
    todo!("C1c: relu")
}

/// C2a. One layer: `A x + b` (one row of `a` per neuron), then the ReLU on every entry
/// if `bend` is true.
///
/// Hint: every row `a[row]` is one neuron: `weighted_sum(&a[row], b[row], x)`.
pub fn layer(a: &[Vec<f32>], b: &[f32], x: &[f32], bend: bool) -> Vec<f32> {
    todo!("C2a: layer")
}

/// C2b. The network of the lecture: `out = C h + d` with `h = ReLU(A x + b)` (or without
/// the ReLU if `bend` is false). One output.
///
/// Hint: the output is one more weighted sum, of `h` with the weights `c` and the shift `d`.
pub fn network(a: &[Vec<f32>], b: &[f32], c: &[f32], d: f32, x: &[f32], bend: bool) -> f32 {
    todo!("C2b: network")
}

/// C3. Softmax: class scores to probabilities that sum to 1. Subtract the largest score
/// first, so that `exp` cannot overflow (slide "The math: softmax").
///
/// Hint: find the largest score `max`; then `p[k] = exp(z[k] - max) / sum of all exp(z[j] - max)`.
pub fn softmax(z: &[f32]) -> Vec<f32> {
    todo!("C3: softmax")
}

/// The mean cross-entropy of one sigmoid neuron on labelled points (labels 0 or 1):
/// the loss that `train_neuron` walks down.
pub fn neuron_loss(w: &[f32], b: f32, points: &[Vec<f32>], labels: &[f32]) -> f32 {
    let mut loss = 0.0;
    for (x, &y) in points.iter().zip(labels) {
        let p = sigmoid(weighted_sum(w, b, x)).clamp(1e-7, 1.0 - 1e-7);
        loss -= y * p.ln() + (1.0 - y) * (1.0 - p).ln();
    }
    loss / points.len() as f32
}

/// C4. Train one sigmoid neuron by gradient descent: start at `w = 0`, `b = 0`, then
/// `steps` times step against the gradient of the mean cross-entropy. For one point the
/// gradient is `(p - y) x` for the weights and `(p - y)` for the bias, with `p` the
/// neuron's output. Returns `(w, b)`.
///
/// Hint: in every step, add up `(p - y) * x[i] / n` over all points for each weight (and
/// `(p - y) / n` for the bias), then subtract `eta` times these sums from `w` and `b`.
pub fn train_neuron(points: &[Vec<f32>], labels: &[f32], eta: f32, steps: usize) -> (Vec<f32>, f32) {
    todo!("C4: train_neuron")
}
