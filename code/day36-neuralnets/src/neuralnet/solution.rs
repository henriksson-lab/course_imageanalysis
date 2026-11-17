// Model solution for the coding exercises of "Neural networks: matrix, bend, matrix".

/// C1a. One neuron before its bend: the weighted sum `w1 x1 + w2 x2 + ... + b`.
/// (Apply `sigmoid` or `relu` to the result to get the neuron's output.)
pub fn weighted_sum(w: &[f32], b: f32, x: &[f32]) -> f32 {
    let mut s = b;
    for i in 0..w.len() {
        s += w[i] * x[i];
    }
    s
}

/// C1b. The sigmoid `1 / (1 + e^(-s))`: a number between 0 and 1.
pub fn sigmoid(s: f32) -> f32 {
    1.0 / (1.0 + (-s).exp())
}

/// C1c. The ReLU `max(0, s)`: negatives become 0.
pub fn relu(s: f32) -> f32 {
    s.max(0.0)
}

/// C2a. One layer: `A x + b` (one row of `a` per neuron), then the ReLU on every entry
/// if `bend` is true.
pub fn layer(a: &[Vec<f32>], b: &[f32], x: &[f32], bend: bool) -> Vec<f32> {
    let mut h = Vec::with_capacity(a.len());
    for row in 0..a.len() {
        let s = weighted_sum(&a[row], b[row], x); // every row of A is one neuron
        h.push(if bend { relu(s) } else { s });
    }
    h
}

/// C2b. The network of the lecture: `out = C h + d` with `h = ReLU(A x + b)` (or without
/// the ReLU if `bend` is false). One output.
pub fn network(a: &[Vec<f32>], b: &[f32], c: &[f32], d: f32, x: &[f32], bend: bool) -> f32 {
    let h = layer(a, b, x, bend);
    weighted_sum(c, d, &h)
}

/// C3. Softmax: class scores to probabilities that sum to 1. Subtract the largest score
/// first, so that `exp` cannot overflow (slide "The math: softmax").
pub fn softmax(z: &[f32]) -> Vec<f32> {
    let max = z.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let e: Vec<f32> = z.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = e.iter().sum();
    e.iter().map(|&v| v / sum).collect()
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
pub fn train_neuron(points: &[Vec<f32>], labels: &[f32], eta: f32, steps: usize) -> (Vec<f32>, f32) {
    let n = points.len() as f32;
    let mut w = vec![0.0; points[0].len()];
    let mut b = 0.0;
    for _ in 0..steps {
        let mut gw = vec![0.0; w.len()];
        let mut gb = 0.0;
        for (x, &y) in points.iter().zip(labels) {
            let err = sigmoid(weighted_sum(&w, b, x)) - y; // p - y
            for i in 0..w.len() {
                gw[i] += err * x[i] / n;
            }
            gb += err / n;
        }
        for i in 0..w.len() {
            w[i] -= eta * gw[i]; // walk downhill
        }
        b -= eta * gb;
    }
    (w, b)
}
