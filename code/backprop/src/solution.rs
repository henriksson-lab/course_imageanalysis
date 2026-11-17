// Model solution for the coding exercises of "Learning: loss, gradient descent and backpropagation".

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
pub fn cross_entropy(p: &[f32], true_class: usize) -> f32 {
    -p[true_class].ln()
}

/// The values of a forward pass: `z` and `h` of every hidden unit, and the output.
#[derive(Clone, Debug)]
pub struct Forward {
    pub z: Vec<f32>,
    pub h: Vec<f32>,
    pub y_hat: f32,
}

/// C2. Forward pass: compute and keep `z`, `h` of every hidden unit and the output `y_hat`.
pub fn forward(net: &Net, x: f32) -> Forward {
    let n = net.w1.len();
    let mut z = vec![0.0; n];
    let mut h = vec![0.0; n];
    let mut y_hat = net.b2;
    for k in 0..n {
        z[k] = net.w1[k] * x + net.b1[k];
        h[k] = z[k].max(0.0);
        y_hat += net.w2[k] * h[k];
    }
    Forward { z, h, y_hat }
}

/// The squared error `(y_hat - y)^2` of one example.
pub fn loss(net: &Net, x: f32, y: f32) -> f32 {
    let e = forward(net, x).y_hat - y;
    e * e
}

/// C3. Backward pass for one example and the loss `(y_hat - y)^2`: the gradient of the
/// loss with respect to every weight, in a `Net`. Each node's gradient is the gradient
/// coming from above times its local derivative (slide "Backprop by hand").
pub fn backward(net: &Net, x: f32, y: f32) -> Net {
    let f = forward(net, x);
    let mut g = Net::zeros(net.w1.len());
    let d_y = 2.0 * (f.y_hat - y); // dL/dy_hat
    g.b2 = d_y;
    for k in 0..net.w1.len() {
        g.w2[k] = d_y * f.h[k];
        let d_h = d_y * net.w2[k];
        let d_z = if f.z[k] > 0.0 { d_h } else { 0.0 }; // the ReLU passes or blocks
        g.w1[k] = d_z * x;
        g.b1[k] = d_z;
    }
    g
}

/// C4. One step of gradient descent on the mean loss over all examples: average the
/// gradients of `backward`, then `weight -= eta * gradient` for every weight.
/// Returns the mean loss **before** the step.
pub fn train_step(net: &mut Net, xs: &[f32], ys: &[f32], eta: f32) -> f32 {
    let n = xs.len() as f32;
    let mut mean = Net::zeros(net.w1.len());
    let mut total = 0.0;
    for (&x, &y) in xs.iter().zip(ys) {
        total += loss(net, x, y);
        let g = backward(net, x, y);
        for k in 0..net.w1.len() {
            mean.w1[k] += g.w1[k] / n;
            mean.b1[k] += g.b1[k] / n;
            mean.w2[k] += g.w2[k] / n;
        }
        mean.b2 += g.b2 / n;
    }
    for k in 0..net.w1.len() {
        net.w1[k] -= eta * mean.w1[k];
        net.b1[k] -= eta * mean.b1[k];
        net.w2[k] -= eta * mean.w2[k];
    }
    net.b2 -= eta * mean.b2;
    total / n
}

/// C5. The same gradient as `backward`, by finite differences: for every weight,
/// `(loss with the weight + eps) - (loss with the weight - eps)`, divided by `2 eps`.
/// One forward pass per weight and side: slow, but a good check.
pub fn numerical_gradient(net: &Net, x: f32, y: f32, eps: f32) -> Net {
    let weights = net.to_vec();
    let mut g = vec![0.0; weights.len()];
    for i in 0..weights.len() {
        let mut plus = weights.clone();
        plus[i] += eps;
        let mut minus = weights.clone();
        minus[i] -= eps;
        let h = net.w1.len();
        g[i] = (loss(&Net::from_vec(h, &plus), x, y) - loss(&Net::from_vec(h, &minus), x, y)) / (2.0 * eps);
    }
    Net::from_vec(net.w1.len(), &g)
}
