// Model solution for the coding exercises of "Transfer learning".

pub use crate::common::*;

/// C1. The features of a patch after the first `layers` layers of a network (each followed by
/// its ReLU): the frozen backbone's description of the patch.
pub fn features(net: &Net, x: &[f32], layers: usize) -> Vec<f32> {
    let mut a = x.to_vec();
    for layer in &net[..layers] {
        a = layer
            .w
            .iter()
            .zip(&layer.b)
            .map(|(row, b)| (b + row.iter().zip(&a).map(|(w, v)| w * v).sum::<f32>()).max(0.0))
            .collect();
    }
    a
}

/// C2. A linear probe: softmax regression on fixed features, `steps` steps of full-batch
/// gradient descent with learning rate `lr`, starting from all weights 0. The gradient of the
/// mean cross-entropy for one example is `(p − y) fᵀ` for the weights and `p − y` for the biases.
pub fn train_probe(feats: &[Vec<f32>], ys: &[usize], k: usize, steps: usize, lr: f32) -> Layer {
    let d = feats[0].len();
    let n = feats.len() as f32;
    let mut probe = Layer { w: vec![vec![0.0; d]; k], b: vec![0.0; k] };
    for _ in 0..steps {
        let mut g = probe.zeros_like();
        for (f, &y) in feats.iter().zip(ys) {
            let p = softmax(&scores(&probe, f));
            for c in 0..k {
                let e = (p[c] - if c == y { 1.0 } else { 0.0 }) / n;
                g.b[c] += e;
                for j in 0..d {
                    g.w[c][j] += e * f[j];
                }
            }
        }
        for c in 0..k {
            probe.b[c] -= lr * g.b[c];
            for j in 0..d {
                probe.w[c][j] -= lr * g.w[c][j];
            }
        }
    }
    probe
}

/// The class scores of a probe: `W f + b`.
pub fn scores(probe: &Layer, f: &[f32]) -> Vec<f32> {
    probe.w.iter().zip(&probe.b).map(|(row, b)| b + row.iter().zip(f).map(|(w, v)| w * v).sum::<f32>()).collect()
}

/// C3. The fraction of examples whose highest probe score is the true class.
pub fn probe_accuracy(probe: &Layer, feats: &[Vec<f32>], ys: &[usize]) -> f32 {
    let ok = feats.iter().zip(ys).filter(|(f, &y)| argmax(&scores(probe, f)) == y).count();
    ok as f32 / feats.len() as f32
}

/// C4. How many weights and biases are trained when the first `frozen` layers are frozen.
pub fn trained_weights(net: &Net, frozen: usize) -> usize {
    net[frozen..].iter().map(|l| l.count()).sum()
}
