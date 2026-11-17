// Coding exercises for "Drifting: generating in one step". Replace each `todo!()` with your code,
// then check it with `cargo test -p diffusion drifting::` and look at `cargo run -p diffusion -- drifting`.
#![allow(unused_variables, unused_imports)]

/// C1. The kernel-weighted mean of the values in `set` as seen from `x`: weights
/// `exp(−|x − s| / tau)`, divided by their sum (a softmax of `−|x − s| / tau`). The entry with
/// index `skip` (if any) is left out. If nothing is left, return `x`.
///
/// Hint: for a small `tau` every weight can underflow to 0. Subtract the smallest distance
/// from every distance before `exp` (the softmax does not change).
pub fn kernel_mean(x: f32, set: &[f32], tau: f32, skip: Option<usize>) -> f32 {
    todo!("C1: kernel_mean")
}

/// C2. The drifting field at every generated point: `V = V⁺ − repel · V⁻` with
/// `V⁺ = kernel_mean(x, real) − x` (the pull towards the real points) and
/// `V⁻ = kernel_mean(x, other generated points) − x` (the push away from the others; a point is
/// never its own negative). `repel = 1` is drifting, `repel = 0` attraction only (mean shift).
pub fn field(generated: &[f32], real: &[f32], tau: f32, repel: f32) -> Vec<f32> {
    todo!("C2: field")
}

/// The distance from `set[i]` to its `k`-th nearest other value in `set`.
pub fn kth_neighbour(set: &[f32], i: usize, k: usize) -> f32 {
    let mut d: Vec<f32> = set.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, s)| (s - set[i]).abs()).collect();
    d.sort_by(|a, b| a.total_cmp(b));
    d[(k - 1).min(d.len() - 1)]
}

/// C3. Coverage: the fraction of real values whose `k`-NN ball (radius = distance to the `k`-th
/// nearest other real value, `kth_neighbour`) contains at least one generated value.
pub fn coverage(real: &[f32], generated: &[f32], k: usize) -> f32 {
    todo!("C3: coverage")
}

/// C4. One drifting step of free points: compute the field at every point (all with the
/// positions before the step), then move every point by `eta · V`. Returns the mean of `|V|`.
pub fn drift_step(points: &mut [f32], real: &[f32], tau: f32, eta: f32, repel: f32) -> f32 {
    todo!("C4: drift_step")
}

/// A small generator `f(ε) = Σ_k w2[k] · tanh(w1[k] · ε + b1[k]) + b2` (one input, one output,
/// `H` hidden units), and its gradients (the same type).
#[derive(Clone, Debug)]
pub struct Net {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: f32,
}

impl Net {
    /// A network with `h` hidden units and small random weights.
    pub fn new(h: usize, rng: &mut imgcourse::Rng) -> Net {
        Net {
            w1: (0..h).map(|_| rng.normal()).collect(),
            b1: (0..h).map(|_| 0.5 * rng.normal()).collect(),
            w2: (0..h).map(|_| rng.normal() / (h as f32).sqrt()).collect(),
            b2: 0.0,
        }
    }

    /// A network of the same size with all weights 0 (to hold gradients).
    pub fn zeros_like(&self) -> Net {
        let h = self.w1.len();
        Net { w1: vec![0.0; h], b1: vec![0.0; h], w2: vec![0.0; h], b2: 0.0 }
    }

    /// The output for one input: one forward pass.
    pub fn forward(&self, eps: f32) -> f32 {
        let mut y = self.b2;
        for k in 0..self.w1.len() {
            y += self.w2[k] * (self.w1[k] * eps + self.b1[k]).tanh();
        }
        y
    }

    /// Backpropagation: add to `grad` the gradient of a loss with respect to every weight, given
    /// the derivative `d_out` of that loss with respect to the output for the input `eps`.
    pub fn backward(&self, eps: f32, d_out: f32, grad: &mut Net) {
        grad.b2 += d_out;
        for k in 0..self.w1.len() {
            let h = (self.w1[k] * eps + self.b1[k]).tanh();
            grad.w2[k] += d_out * h;
            let d_z = d_out * self.w2[k] * (1.0 - h * h);
            grad.w1[k] += d_z * eps;
            grad.b1[k] += d_z;
        }
    }

    /// Gradient descent: every weight minus `lr` times its gradient.
    pub fn step(&mut self, grad: &Net, lr: f32) {
        for k in 0..self.w1.len() {
            self.w1[k] -= lr * grad.w1[k];
            self.b1[k] -= lr * grad.b1[k];
            self.w2[k] -= lr * grad.w2[k];
        }
        self.b2 -= lr * grad.b2;
    }
}

/// C5. One training step of the generator with the drifting loss
/// `L = mean over the batch of (f(ε) − stopgrad(f(ε) + V))²`:
/// samples `f(ε)` for the noise values `eps`, the field `V` at the samples (positives: `real_batch`,
/// negatives: the other samples), `dL/df = −2V / batch size` for every sample, backpropagation,
/// one gradient-descent step with `lr`. Returns the loss (the mean of `V²`).
///
/// Hint: `net.zeros_like()` for the gradient, `net.backward(e, d_out, &mut grad)` for every sample,
/// then `net.step(&grad, lr)`. The field is computed once, before the step, and not differentiated.
pub fn train_step(net: &mut Net, eps: &[f32], real_batch: &[f32], tau: f32, lr: f32, repel: f32) -> f32 {
    todo!("C5: train_step")
}
