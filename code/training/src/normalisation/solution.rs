// Model solution for "Initialisation, normalisation and residual connections".

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
pub fn init_weights(fan_in: usize, fan_out: usize, gain: f32, rng: &mut Rng) -> Vec<f32> {
    let sd = gain / (fan_in as f32).sqrt();
    (0..fan_in * fan_out).map(|_| sd * rng.normal()).collect()
}

/// C2. Batch normalisation (training mode, no learned scale and shift): for every unit j,
/// subtract the mean of `z[n][j]` over the examples n of the batch and divide by
/// `sqrt(variance + eps)`. Returns the means and variances it used (one per unit).
pub fn batch_norm(z: &mut [Vec<f32>], eps: f32) -> (Vec<f32>, Vec<f32>) {
    let (n, units) = (z.len() as f32, z[0].len());
    let mut means = vec![0.0; units];
    let mut vars = vec![0.0; units];
    for j in 0..units {
        let mean = z.iter().map(|row| row[j]).sum::<f32>() / n;
        let var = z.iter().map(|row| (row[j] - mean).powi(2)).sum::<f32>() / n;
        for row in z.iter_mut() {
            row[j] = (row[j] - mean) / (var + eps).sqrt();
        }
        means[j] = mean;
        vars[j] = var;
    }
    (means, vars)
}

/// C3. Layer normalisation of one example: subtract the mean of its values `z[j]` over the
/// units j and divide by `sqrt(variance + eps)`. Needs no batch: the same at training and test time.
pub fn layer_norm(z: &mut [f32], eps: f32) {
    let n = z.len() as f32;
    let mean = z.iter().sum::<f32>() / n;
    let var = z.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    for v in z.iter_mut() {
        *v = (*v - mean) / (var + eps).sqrt();
    }
}

/// C4. Push a batch of `batch` random inputs (normal, mean 0, sd 1, `width` numbers each)
/// through `depth` dense layers of `width` units, with weights from `init_weights(.., gain, ..)`,
/// zero biases, the normalisation `norm` (eps 1e-5) on each layer's sums and then the bend
/// `act`. Returns the root mean square of the layer's outputs over the whole batch, one number per
/// layer: how large the signal is after layer 1, 2, ..., depth.
pub fn signal_sizes(depth: usize, width: usize, gain: f32, act: Act, norm: Norm, batch: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut a: Vec<Vec<f32>> = (0..batch).map(|_| (0..width).map(|_| rng.normal()).collect()).collect();
    let mut sizes = Vec::new();
    for _ in 0..depth {
        let w = init_weights(width, width, gain, &mut rng);
        let mut z: Vec<Vec<f32>> = a
            .iter()
            .map(|x| (0..width).map(|j| (0..width).map(|i| w[j * width + i] * x[i]).sum()).collect())
            .collect();
        match norm {
            Norm::None => {}
            Norm::Batch => {
                batch_norm(&mut z, 1e-5);
            }
            Norm::Layer => z.iter_mut().for_each(|row| layer_norm(row, 1e-5)),
        }
        a = z
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&v| match act {
                        Act::Linear => v,
                        Act::Relu => v.max(0.0),
                        Act::Tanh => v.tanh(),
                    })
                    .collect()
            })
            .collect();
        let ms = a.iter().flatten().map(|v| v * v).sum::<f32>() / (batch * width) as f32;
        sizes.push(ms.sqrt());
    }
    sizes
}
