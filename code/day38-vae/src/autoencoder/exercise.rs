// Coding exercises for "From PCA to autoencoders". Replace each `todo!()` with your code,
// then check it with `cargo test -p day38 autoencoder::` and look at `cargo run -p day38 -- autoencoder`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Rng;

/// C1a. Encode with PCA: the score `s_k = u_k · (x − mean)` for every component `u_k`
/// (unit vectors). Returns one number per component.
///
/// Hint: one loop over the components, and inside it a sum over the pixels `i`.
pub fn pca_encode(x: &[f32], mean: &[f32], components: &[Vec<f32>]) -> Vec<f32> {
    todo!("C1a: pca_encode")
}

/// C1b. Decode with PCA: `mean + s_1 u_1 + ... + s_k u_k`.
///
/// Hint: start from `mean.to_vec()` and add `s * u[i]` to every pixel, for every pair of score
/// and component (`scores.iter().zip(components)`).
pub fn pca_decode(scores: &[f32], mean: &[f32], components: &[Vec<f32>]) -> Vec<f32> {
    todo!("C1b: pca_decode")
}

/// C2. The mean squared reconstruction error of a set of images: encode and decode every
/// image, and average the squared length `|x − x̂|²` of what is lost.
///
/// Hint: use `pca_encode` and `pca_decode`, add up `(x[i] - rebuilt[i]).powi(2)` over all
/// pixels of all images, divide by the number of images.
pub fn reconstruction_error(images: &[Vec<f32>], mean: &[f32], components: &[Vec<f32>]) -> f32 {
    todo!("C2: reconstruction_error")
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
///
/// Hint: start from a copy of the shifts, `layer.b.clone()`, and add `layer.w[o][i] * input[i]`
/// for every output `o` and input `i`; then `max(0.0)` if `relu`.
pub fn dense(layer: &Layer, input: &[f32], relu: bool) -> Vec<f32> {
    todo!("C3a: dense")
}

/// C3b. Run the autoencoder on one image: returns every layer's output,
/// `[h1, z, h2, x̂]` (16, 1, 16 and 2 numbers). `z` is the code, `x̂` the reconstruction.
///
/// Hint: four calls of `dense`, each on the output of the one before; ReLU on the first and the
/// third.
pub fn forward(ae: &Autoencoder, x: &[f32]) -> [Vec<f32>; 4] {
    todo!("C3b: forward")
}

/// C4. One step of gradient descent on the mean squared reconstruction error of `images`
/// (backpropagation, as in the unit *Neural networks*). Changes the weights of `ae` and
/// returns the error before the step.
///
/// Hint: for every image run `forward` and keep the four outputs; the input of layer k is the
/// image for k = 0, else the output of layer k − 1. Start with the gradient at the output, `2
/// (x̂ − x) / n`. Then for k = 3, 2, 1, 0: add `g[o] * input[i]` to the gradient of `w[o][i]`
/// (and `g[o]` to that of `b[o]`); pass the gradient back to the input, `back[i] = Σ_o w[o][i]
/// g[o]`; for k = 3 and k = 1 the input came out of a ReLU, so set `back[i] = 0` where the
/// input is 0. Add the gradients over all images (in `Layer::zeros_like` copies) and only then
/// change the weights: `w -= rate * gradient`.
pub fn train_step(ae: &mut Autoencoder, images: &[Vec<f32>], rate: f32) -> f32 {
    todo!("C4: train_step")
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
