//! A deep, narrow network with or without residual connections, for the program of this unit
//! (provided code, not an exercise). Input 2 → `width` units; then `blocks` blocks, each
//! `z = W relu(h) + b`; a plain block replaces h by z, a residual block adds it: h + z; then
//! scores = W_out relu(h) + b_out for 3 classes, softmax and cross-entropy. Trained with Adam.

use crate::common::{argmax, shuffled, softmax, Adam};
use imgcourse::Rng;

pub struct DeepNet {
    pub width: usize,
    pub blocks: usize,
    pub residual: bool,
    /// All weights in one list: input layer (W, b), every block (W, b), output layer (W, b).
    pub params: Vec<f32>,
}

impl DeepNet {
    /// He initialisation (sd sqrt(2 / fan-in)); in a residual network the weights of every block
    /// are also multiplied by 1/sqrt(blocks), so that each block starts as a small change.
    pub fn new(width: usize, blocks: usize, residual: bool, seed: u64) -> DeepNet {
        let mut rng = Rng::new(seed);
        let mut params = Vec::new();
        let mut layer = |n_in: usize, n_out: usize, scale: f32, params: &mut Vec<f32>| {
            let sd = scale * (2.0 / n_in as f32).sqrt();
            for _ in 0..n_in * n_out {
                params.push(sd * rng.normal());
            }
            params.extend(std::iter::repeat_n(0.0, n_out));
        };
        layer(2, width, 1.0, &mut params);
        let s = if residual { 1.0 / (blocks as f32).sqrt() } else { 1.0 };
        for _ in 0..blocks {
            layer(width, width, s, &mut params);
        }
        layer(width, 3, 1.0, &mut params);
        DeepNet { width, blocks, residual, params }
    }

    /// (offset, n_in, n_out) of layer k: 0 the input layer, 1..=blocks the blocks, then the output.
    fn layer(&self, k: usize) -> (usize, usize, usize) {
        let w = self.width;
        let first = 2 * w + w;
        let block = w * w + w;
        if k == 0 {
            (0, 2, w)
        } else if k <= self.blocks {
            (first + (k - 1) * block, w, w)
        } else {
            (first + self.blocks * block, w, 3)
        }
    }

    fn affine(&self, k: usize, x: &[f32]) -> Vec<f32> {
        let (o, n_in, n_out) = self.layer(k);
        (0..n_out).map(|j| self.params[o + n_in * n_out + j] + (0..n_in).map(|i| self.params[o + j * n_in + i] * x[i]).sum::<f32>()).collect()
    }

    /// The states h_0 … h_blocks and the scores.
    fn forward(&self, x: &[f32]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let relu = |v: &[f32]| v.iter().map(|a| a.max(0.0)).collect::<Vec<f32>>();
        let mut hs = vec![self.affine(0, x)];
        for l in 1..=self.blocks {
            let z = self.affine(l, &relu(&hs[l - 1]));
            let h = if self.residual { hs[l - 1].iter().zip(&z).map(|(a, b)| a + b).collect() } else { z };
            hs.push(h);
        }
        let scores = self.affine(self.blocks + 1, &relu(&hs[self.blocks]));
        (hs, scores)
    }

    /// Mean cross-entropy and its gradient over the examples `idx`; also the root mean square of
    /// dL/dh at every state h_0 … h_blocks (summed over the batch before squaring).
    pub fn gradient(&self, xs: &[Vec<f32>], ys: &[usize], idx: &[usize]) -> (f32, Vec<f32>, Vec<f32>) {
        let mut g = vec![0.0; self.params.len()];
        let mut dh_sum = vec![vec![0.0f32; self.width]; self.blocks + 1];
        let m = idx.len() as f32;
        let mut loss = 0.0;
        let relu = |v: &[f32]| v.iter().map(|a| a.max(0.0)).collect::<Vec<f32>>();
        // adds dz ⊗ input to the gradient of layer k and returns W^T dz
        let back = |k: usize, dz: &[f32], input: &[f32], g: &mut Vec<f32>| -> Vec<f32> {
            let (o, n_in, n_out) = self.layer(k);
            let mut d_in = vec![0.0; n_in];
            for j in 0..n_out {
                g[o + n_in * n_out + j] += dz[j];
                for i in 0..n_in {
                    g[o + j * n_in + i] += dz[j] * input[i];
                    d_in[i] += self.params[o + j * n_in + i] * dz[j];
                }
            }
            d_in
        };
        for &n in idx {
            let (hs, scores) = self.forward(&xs[n]);
            let p = softmax(&scores);
            loss -= p[ys[n]].max(1e-12).ln();
            let ds: Vec<f32> = (0..3).map(|k| (p[k] - if k == ys[n] { 1.0 } else { 0.0 }) / m).collect();
            let top = &hs[self.blocks];
            let du = back(self.blocks + 1, &ds, &relu(top), &mut g);
            let mut dh: Vec<f32> = du.iter().zip(top).map(|(d, h)| if *h > 0.0 { *d } else { 0.0 }).collect();
            for l in (1..=self.blocks).rev() {
                for i in 0..self.width {
                    dh_sum[l][i] += dh[i];
                }
                let below = &hs[l - 1];
                let du = back(l, &dh, &relu(below), &mut g);
                let through: Vec<f32> = du.iter().zip(below).map(|(d, h)| if *h > 0.0 { *d } else { 0.0 }).collect();
                dh = if self.residual { through.iter().zip(&dh).map(|(a, b)| a + b).collect() } else { through };
            }
            for i in 0..self.width {
                dh_sum[0][i] += dh[i];
            }
            back(0, &dh, &xs[n], &mut g);
        }
        let sizes = dh_sum.iter().map(|v| (v.iter().map(|a| a * a).sum::<f32>() / v.len() as f32).sqrt()).collect();
        (loss / m, g, sizes)
    }

    pub fn loss_and_accuracy(&self, xs: &[Vec<f32>], ys: &[usize]) -> (f32, f32) {
        let (mut l, mut ok) = (0.0, 0.0);
        for (x, &y) in xs.iter().zip(ys) {
            let s = self.forward(x).1;
            l -= softmax(&s)[y].max(1e-12).ln();
            if argmax(&s) == y {
                ok += 1.0;
            }
        }
        (l / xs.len() as f32, ok / xs.len() as f32)
    }
}

/// Trains for `epochs` epochs (Adam, learning rate `eta`, batches of 32); returns the training
/// loss after every epoch (epoch 0 first) and the gradient sizes of the very first batch.
pub fn train_deep(net: &mut DeepNet, xs: &[Vec<f32>], ys: &[usize], epochs: usize, eta: f32) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(1);
    let mut adam = Adam::new(net.params.len());
    let mut curve = vec![net.loss_and_accuracy(xs, ys).0];
    let mut first = Vec::new();
    for _ in 0..epochs {
        let order = shuffled(xs.len(), &mut rng);
        for batch in order.chunks(32) {
            let (_, g, sizes) = net.gradient(xs, ys, batch);
            if first.is_empty() {
                first = sizes;
            }
            adam.step(&mut net.params, &g, eta);
        }
        curve.push(net.loss_and_accuracy(xs, ys).0);
    }
    (curve, first)
}
