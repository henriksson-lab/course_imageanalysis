// Model solution for the coding exercises of "From PCA to autoencoders".

use imgcourse::Rng;

/// C1a. Encode with PCA: the score `s_k = u_k · (x − mean)` for every component `u_k`
/// (unit vectors). Returns one number per component.
pub fn pca_encode(x: &[f32], mean: &[f32], components: &[Vec<f32>]) -> Vec<f32> {
    let mut scores = Vec::new();
    for u in components {
        let mut s = 0.0;
        for i in 0..x.len() {
            s += u[i] * (x[i] - mean[i]);
        }
        scores.push(s);
    }
    scores
}

/// C1b. Decode with PCA: `mean + s_1 u_1 + ... + s_k u_k`.
pub fn pca_decode(scores: &[f32], mean: &[f32], components: &[Vec<f32>]) -> Vec<f32> {
    let mut x = mean.to_vec();
    for (s, u) in scores.iter().zip(components) {
        for i in 0..x.len() {
            x[i] += s * u[i];
        }
    }
    x
}

/// C2. The mean squared reconstruction error of a set of images: encode and decode every
/// image, and average the squared length `|x − x̂|²` of what is lost.
pub fn reconstruction_error(images: &[Vec<f32>], mean: &[f32], components: &[Vec<f32>]) -> f32 {
    let mut total = 0.0;
    for x in images {
        let rebuilt = pca_decode(&pca_encode(x, mean, components), mean, components);
        for i in 0..x.len() {
            total += (x[i] - rebuilt[i]).powi(2);
        }
    }
    total / images.len() as f32
}

/// One layer of a network: `output = w · input + b`, with `w[out][in]`.
#[derive(Clone, Debug)]
pub struct Layer {
    pub w: Vec<Vec<f32>>,
    pub b: Vec<f32>,
}

impl Layer {
    /// Small random weights (fixed seed), zero shifts.
    pub fn random(inputs: usize, outputs: usize, rng: &mut Rng) -> Layer {
        let scale = (2.0 / inputs as f32).sqrt();
        let w = (0..outputs).map(|_| (0..inputs).map(|_| scale * rng.normal()).collect()).collect();
        Layer { w, b: vec![0.0; outputs] }
    }

    /// A layer of the same size with all weights and shifts 0 (to add up gradients in).
    pub fn zeros_like(&self) -> Layer {
        Layer { w: vec![vec![0.0; self.w[0].len()]; self.w.len()], b: vec![0.0; self.b.len()] }
    }
}

/// The autoencoder of the lecture: encoder 2 → 16 → 1, decoder 1 → 16 → 2, with ReLU after
/// the two hidden layers of 16 (and none on the code `z` or on the output).
#[derive(Clone, Debug)]
pub struct Autoencoder {
    pub layers: [Layer; 4],
}

impl Autoencoder {
    pub fn new(hidden: usize, seed: u64) -> Autoencoder {
        let mut rng = Rng::new(seed);
        Autoencoder {
            layers: [
                Layer::random(2, hidden, &mut rng),
                Layer::random(hidden, 1, &mut rng),
                Layer::random(1, hidden, &mut rng),
                Layer::random(hidden, 2, &mut rng),
            ],
        }
    }
}

/// C3a. One layer: `w · input + b`, then the bend `max(0, ·)` if `relu` is true.
pub fn dense(layer: &Layer, input: &[f32], relu: bool) -> Vec<f32> {
    let mut out = layer.b.clone();
    for o in 0..out.len() {
        for i in 0..input.len() {
            out[o] += layer.w[o][i] * input[i];
        }
        if relu {
            out[o] = out[o].max(0.0);
        }
    }
    out
}

/// C3b. Run the autoencoder on one image: returns every layer's output,
/// `[h1, z, h2, x̂]` (16, 1, 16 and 2 numbers). `z` is the code, `x̂` the reconstruction.
pub fn forward(ae: &Autoencoder, x: &[f32]) -> [Vec<f32>; 4] {
    let h1 = dense(&ae.layers[0], x, true);
    let z = dense(&ae.layers[1], &h1, false);
    let h2 = dense(&ae.layers[2], &z, true);
    let xhat = dense(&ae.layers[3], &h2, false);
    [h1, z, h2, xhat]
}

/// C4. One step of gradient descent on the mean squared reconstruction error of `images`
/// (backpropagation, as in the unit *Neural networks*). Changes the weights of `ae` and
/// returns the error before the step.
pub fn train_step(ae: &mut Autoencoder, images: &[Vec<f32>], rate: f32) -> f32 {
    let n = images.len() as f32;
    let mut grads: Vec<Layer> = ae.layers.iter().map(|l| l.zeros_like()).collect();
    let mut loss = 0.0;
    for x in images {
        let outs = forward(ae, x);
        // the input of layer k is x for k = 0, else the output of layer k − 1
        let inputs = [x.as_slice(), &outs[0], &outs[1], &outs[2]];
        // d(error)/d(x̂) for this image's share of the mean error
        let mut g: Vec<f32> = (0..2).map(|i| 2.0 * (outs[3][i] - x[i]) / n).collect();
        loss += (0..2).map(|i| (outs[3][i] - x[i]).powi(2)).sum::<f32>() / n;
        for k in (0..4).rev() {
            let layer = &ae.layers[k];
            let input = inputs[k];
            // the gradient of each weight is (gradient at its output) × (its input)
            for o in 0..g.len() {
                grads[k].b[o] += g[o];
                for i in 0..input.len() {
                    grads[k].w[o][i] += g[o] * input[i];
                }
            }
            // pass the gradient back to the input; through a ReLU only where it was > 0
            let mut back = vec![0.0; input.len()];
            for i in 0..input.len() {
                for o in 0..g.len() {
                    back[i] += layer.w[o][i] * g[o];
                }
            }
            if k == 1 || k == 3 {
                // the input of layers 1 and 3 is the output of a ReLU layer
                for i in 0..back.len() {
                    if input[i] <= 0.0 {
                        back[i] = 0.0;
                    }
                }
            }
            g = back;
        }
    }
    for (layer, grad) in ae.layers.iter_mut().zip(&grads) {
        for o in 0..layer.b.len() {
            layer.b[o] -= rate * grad.b[o];
            for i in 0..layer.w[o].len() {
                layer.w[o][i] -= rate * grad.w[o][i];
            }
        }
    }
    loss
}

/// The lecture's moving spot: a bright spot (a Gaussian of sd 0.5 pixel) at `n` positions
/// from −0.5 to 1.5, seen by two pixels at 0 and 1, plus noise of sd `noise`.
pub fn moving_spot(n: usize, noise: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|k| {
            let pos = -0.5 + 2.0 * k as f32 / (n - 1) as f32;
            (0..2).map(|i| (-(pos - i as f32).powi(2) / 0.5).exp() + noise * rng.normal()).collect()
        })
        .collect()
}
