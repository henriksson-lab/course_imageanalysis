// Coding exercises for "Learning: loss, gradient descent and backpropagation". Replace each `todo!()` with your code,
// then check it with `cargo test -p neuralnets backprop::` and look at `cargo run -p neuralnets -- backprop`.
#![allow(unused_variables, unused_imports)]

/// A network with one input, `H` hidden ReLU units and one output (the line-scan network of
/// the lecture): `z_k = w1[k] x + b1[k]`, `h_k = ReLU(z_k)`, `y_hat = sum_k w2[k] h_k + b2`.
/// The same struct also holds gradients: one number per weight.
#[derive(Clone, Debug, PartialEq)]
pub struct Net {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: f32,
}

impl Net {
    /// A network with `h` hidden units whose hinges sit at evenly spread positions in
    /// [0, 1], rising and falling in turn; small output weights.
    pub fn new(h: usize) -> Net {
        let mut net = Net::zeros(h);
        for k in 0..h {
            let position = (k as f32 + 0.5) / h as f32;
            let slope = if k % 2 == 0 { 4.0 } else { -4.0 };
            net.w1[k] = slope;
            net.b1[k] = -slope * position; // the hinge bends where w1 x + b1 = 0
            net.w2[k] = 0.02;
        }
        net
    }

    /// All weights 0 (used for gradients).
    pub fn zeros(h: usize) -> Net {
        Net { w1: vec![0.0; h], b1: vec![0.0; h], w2: vec![0.0; h], b2: 0.0 }
    }

    /// All weights in one list: w1, then b1, then w2, then b2.
    pub fn to_vec(&self) -> Vec<f32> {
        let mut v = Vec::new();
        v.extend(&self.w1);
        v.extend(&self.b1);
        v.extend(&self.w2);
        v.push(self.b2);
        v
    }

    /// The network with `h` hidden units whose weights are the list `v` (as `to_vec`).
    pub fn from_vec(h: usize, v: &[f32]) -> Net {
        Net { w1: v[0..h].to_vec(), b1: v[h..2 * h].to_vec(), w2: v[2 * h..3 * h].to_vec(), b2: v[3 * h] }
    }
}

/// C1. Cross-entropy: minus the log of the probability given to the true class.
///
/// Hint: `p[true_class].ln()` is the natural logarithm.
pub fn cross_entropy(p: &[f32], true_class: usize) -> f32 {
    todo!("C1: cross_entropy")
}

/// The values of a forward pass: `z` and `h` of every hidden unit, and the output.
#[derive(Clone, Debug)]
pub struct Forward {
    pub z: Vec<f32>,
    pub h: Vec<f32>,
    pub y_hat: f32,
}

/// C2. Forward pass: compute and keep `z`, `h` of every hidden unit and the output `y_hat`.
///
/// Hint: start `y_hat` at `net.b2`; for every unit k: `z[k] = w1[k] x + b1[k]`,
/// `h[k] = max(0, z[k])`, and add `w2[k] h[k]` to `y_hat`.
pub fn forward(net: &Net, x: f32) -> Forward {
    todo!("C2: forward")
}

/// The squared error `(y_hat - y)^2` of one example.
pub fn loss(net: &Net, x: f32, y: f32) -> f32 {
    let e = forward(net, x).y_hat - y;
    e * e
}

/// C3. Backward pass for one example and the loss `(y_hat - y)^2`: the gradient of the
/// loss with respect to every weight, in a `Net`. Each node's gradient is the gradient
/// coming from above times its local derivative (slide "Backprop by hand").
///
/// Hint: run `forward` first and keep its values. Then `d_y = 2 (y_hat - y)`; the output
/// weights get `d_y h[k]` and `d_y`; `d_h = d_y w2[k]`; `d_z = d_h` if `z[k] > 0`, else 0; the
/// first-layer weights get `d_z x` and `d_z`. `Net::zeros(h)` makes an empty gradient.
pub fn backward(net: &Net, x: f32, y: f32) -> Net {
    todo!("C3: backward")
}

/// C4. One step of gradient descent on the mean loss over all examples: average the
/// gradients of `backward`, then `weight -= eta * gradient` for every weight.
/// Returns the mean loss **before** the step.
///
/// Hint: add up `backward(net, x, y)` divided by the number of examples into a
/// `Net::zeros(..)`, and the losses; update the weights only after the loop.
pub fn train_step(net: &mut Net, xs: &[f32], ys: &[f32], eta: f32) -> f32 {
    todo!("C4: train_step")
}

/// C5. The same gradient as `backward`, by finite differences: for every weight,
/// `(loss with the weight + eps) - (loss with the weight - eps)`, divided by `2 eps`.
/// One forward pass per weight and side: slow, but a good check.
///
/// Hint: `net.to_vec()` gives all weights in one list and `Net::from_vec(h, &list)` turns a list
/// back into a network, so one loop over the list can nudge each weight in turn.
pub fn numerical_gradient(net: &Net, x: f32, y: f32, eps: f32) -> Net {
    todo!("C5: numerical_gradient")
}
