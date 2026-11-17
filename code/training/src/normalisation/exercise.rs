// Coding exercises for "Initialisation, normalisation and residual connections". Replace each `todo!()` with your
// code, then check it with `cargo test -p training normalisation::` and look at
// `cargo run -p training -- normalisation`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Rng;

/// The bend after each layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Act {
    /// No bend: the stack is one big matrix.
    Linear,
    Relu,
    Tanh,
}

/// Normalisation of each layer's sums `z`, before the bend.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Norm {
    None,
    /// Each unit over the examples of the batch.
    Batch,
    /// Each example over the units of the layer.
    Layer,
}

/// C1. Random initial weights of a dense layer with `fan_in` inputs and `fan_out` outputs, row
/// by row (one row of `fan_in` weights per output): normal with mean 0 and standard deviation
/// `gain / sqrt(fan_in)`. Gain 1 keeps the size of the signal through a linear layer, gain
/// sqrt(2) through a ReLU layer.
///
/// Hint: `rng.normal()` has standard deviation 1; collect `fan_in * fan_out` numbers.
pub fn init_weights(fan_in: usize, fan_out: usize, gain: f32, rng: &mut Rng) -> Vec<f32> {
    todo!("C1: init_weights")
}

/// C2. Batch normalisation (training mode, no learned scale and shift): for every unit j,
/// subtract the mean of `z[n][j]` over the examples n of the batch and divide by
/// `sqrt(variance + eps)`. Returns the means and variances it used (one per unit).
///
/// Hint: loop over the units j; inside, one pass over the rows for the mean, one for the
/// variance (divide by the number of examples, not one less), one to normalise.
pub fn batch_norm(z: &mut [Vec<f32>], eps: f32) -> (Vec<f32>, Vec<f32>) {
    todo!("C2: batch_norm")
}

/// C3. Layer normalisation of one example: subtract the mean of its values `z[j]` over the
/// units j and divide by `sqrt(variance + eps)`. Needs no batch: the same at training and test time.
///
/// Hint: the same three passes as `batch_norm`, along the other direction.
pub fn layer_norm(z: &mut [f32], eps: f32) {
    todo!("C3: layer_norm")
}

/// C4. Push a batch of `batch` random inputs (normal, mean 0, sd 1, `width` numbers each)
/// through `depth` dense layers of `width` units, with weights from `init_weights(.., gain, ..)`,
/// zero biases, the normalisation `norm` (eps 1e-5) on each layer's sums and then the bend
/// `act`. Returns the root mean square of the layer's outputs over the whole batch, one number per
/// layer: how large the signal is after layer 1, 2, ..., depth.
///
/// Hint: one `Rng::new(seed)` for the inputs and then the weights of every layer. For each
/// layer: the sums `z[n][j] = sum over i of w[j * width + i] * a[n][i]`, then the norm, then
/// the bend (`v.max(0.0)`, `v.tanh()`); record sqrt(mean of a^2).
pub fn signal_sizes(depth: usize, width: usize, gain: f32, act: Act, norm: Norm, batch: usize, seed: u64) -> Vec<f32> {
    todo!("C4: signal_sizes")
}
