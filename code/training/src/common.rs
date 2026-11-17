//! Helpers shared by the units of the day: a small network (dense layers, ReLU, softmax and
//! cross-entropy) trained with Adam, the two toy data sets (cell crops and three-armed
//! spirals), and line plots of training curves. Not part of the exercises: read it if you like.
//! (The optimisers unit asks you to write the Adam step yourself; this file has its own copy,
//! so that the other units run before you have.)

use imgcourse::{save_rgb, Image, Rng};

// ---------------------------------------------------------------------------------------
// A small network: dense layers with ReLU between them, softmax at the end.

/// A network with layer sizes `sizes` (input, hidden..., classes). All weights and biases are in
/// one flat list `params`: for every layer, its weight matrix (row by row: one row per output),
/// then its biases. A gradient is a list of the same length.
#[derive(Clone, Debug)]
pub struct Mlp {
    pub sizes: Vec<usize>,
    pub params: Vec<f32>,
}

impl Mlp {
    /// Random weights with standard deviation sqrt(2 / fan-in) (the ReLU rule of the
    /// normalisation unit), biases 0. The same seed gives the same network.
    pub fn new(sizes: &[usize], seed: u64) -> Mlp {
        let mut rng = Rng::new(seed);
        let mut params = Vec::new();
        for l in 0..sizes.len() - 1 {
            let sd = (2.0 / sizes[l] as f32).sqrt();
            for _ in 0..sizes[l] * sizes[l + 1] {
                params.push(sd * rng.normal());
            }
            params.extend(std::iter::repeat_n(0.0, sizes[l + 1]));
        }
        Mlp { sizes: sizes.to_vec(), params }
    }

    /// Where layer `l`'s weights start in `params` (its biases follow them).
    fn offset(&self, l: usize) -> usize {
        (0..l).map(|k| self.sizes[k] * self.sizes[k + 1] + self.sizes[k + 1]).sum()
    }

    /// The values of every layer for one input: `a[0]` is the input, the last is the scores
    /// (logits, before the softmax); hidden layers are after the ReLU.
    pub fn forward(&self, x: &[f32]) -> Vec<Vec<f32>> {
        let layers = self.sizes.len() - 1;
        let mut a = vec![x.to_vec()];
        for l in 0..layers {
            let (n_in, n_out, o) = (self.sizes[l], self.sizes[l + 1], self.offset(l));
            let mut z = vec![0.0; n_out];
            for j in 0..n_out {
                let mut s = self.params[o + n_in * n_out + j];
                for i in 0..n_in {
                    s += self.params[o + j * n_in + i] * a[l][i];
                }
                z[j] = if l + 1 < layers { s.max(0.0) } else { s };
            }
            a.push(z);
        }
        a
    }

    /// Class probabilities for one input (softmax of the scores).
    pub fn probs(&self, x: &[f32]) -> Vec<f32> {
        softmax(self.forward(x).last().unwrap())
    }

    /// Mean loss and its gradient over the examples `idx` of `(xs, ys)`: cross-entropy, each
    /// example multiplied by the weight of its class (`class_weight`, all 1 for the plain loss).
    /// The gradient also has the weight decay `decay * sum of w^2` over the weights (not the
    /// biases); the loss returned is without it.
    pub fn gradient(&self, xs: &[Vec<f32>], ys: &[usize], idx: &[usize], class_weight: &[f32], decay: f32) -> (f32, Vec<f32>) {
        let layers = self.sizes.len() - 1;
        let mut g = vec![0.0; self.params.len()];
        let mut loss = 0.0;
        let m = idx.len() as f32;
        for &n in idx {
            let a = self.forward(&xs[n]);
            let p = softmax(&a[layers]);
            let w = class_weight[ys[n]];
            loss += -w * p[ys[n]].max(1e-12).ln();
            // dL/dz of the scores: w (p - one-hot) / m
            let mut d: Vec<f32> = (0..p.len()).map(|k| w * (p[k] - if k == ys[n] { 1.0 } else { 0.0 }) / m).collect();
            for l in (0..layers).rev() {
                let (n_in, n_out, o) = (self.sizes[l], self.sizes[l + 1], self.offset(l));
                let mut d_prev = vec![0.0; n_in];
                for j in 0..n_out {
                    g[o + n_in * n_out + j] += d[j];
                    for i in 0..n_in {
                        g[o + j * n_in + i] += d[j] * a[l][i];
                        d_prev[i] += d[j] * self.params[o + j * n_in + i];
                    }
                }
                if l > 0 {
                    // through the ReLU of layer l: it passed where its output is positive
                    for i in 0..n_in {
                        if a[l][i] <= 0.0 {
                            d_prev[i] = 0.0;
                        }
                    }
                }
                d = d_prev;
            }
        }
        loss /= m;
        if decay > 0.0 {
            for l in 0..layers {
                let (o, k) = (self.offset(l), self.sizes[l] * self.sizes[l + 1]);
                for i in o..o + k {
                    g[i] += 2.0 * decay * self.params[i];
                }
            }
        }
        (loss, g)
    }

    /// Mean cross-entropy over a data set (no weights, no decay).
    pub fn mean_loss(&self, xs: &[Vec<f32>], ys: &[usize]) -> f32 {
        let s: f32 = xs.iter().zip(ys).map(|(x, &y)| -self.probs(x)[y].max(1e-12).ln()).sum();
        s / xs.len() as f32
    }

    /// Mean cross-entropy and share of correctly classified examples, in one pass.
    pub fn loss_and_accuracy(&self, xs: &[Vec<f32>], ys: &[usize]) -> (f32, f32) {
        let (mut loss, mut ok) = (0.0, 0.0);
        for (x, &y) in xs.iter().zip(ys) {
            let p = self.probs(x);
            loss -= p[y].max(1e-12).ln();
            if argmax(&p) == y {
                ok += 1.0;
            }
        }
        (loss / xs.len() as f32, ok / xs.len() as f32)
    }

    /// The class with the largest probability.
    pub fn predict(&self, x: &[f32]) -> usize {
        argmax(self.forward(x).last().unwrap())
    }

    /// Share of correctly classified examples.
    pub fn accuracy(&self, xs: &[Vec<f32>], ys: &[usize]) -> f32 {
        let ok = xs.iter().zip(ys).filter(|(x, &y)| self.predict(x) == y).count();
        ok as f32 / xs.len() as f32
    }

    /// Recall of every class: of the examples of class k, the share predicted as k.
    pub fn recalls(&self, xs: &[Vec<f32>], ys: &[usize], classes: usize) -> Vec<f32> {
        let mut hit = vec![0.0; classes];
        let mut all = vec![0.0; classes];
        for (x, &y) in xs.iter().zip(ys) {
            all[y] += 1.0;
            if self.predict(x) == y {
                hit[y] += 1.0;
            }
        }
        (0..classes).map(|k| if all[k] > 0.0 { hit[k] / all[k] } else { f32::NAN }).collect()
    }
}

/// Softmax: exp of each score, divided by their sum (the largest score subtracted first).
pub fn softmax(z: &[f32]) -> Vec<f32> {
    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

/// Index of the largest value.
pub fn argmax(v: &[f32]) -> usize {
    (0..v.len()).fold(0, |b, k| if v[k] > v[b] { k } else { b })
}

/// Adam (the optimisers unit): running means of the gradient and of its square, with bias
/// correction; beta1 = 0.9, beta2 = 0.999, eps = 1e-8.
#[derive(Clone, Debug)]
pub struct Adam {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub t: i32,
}

impl Adam {
    pub fn new(n: usize) -> Adam {
        Adam { m: vec![0.0; n], v: vec![0.0; n], t: 0 }
    }

    /// One step on the weights `w` with gradient `g` and learning rate `eta`.
    pub fn step(&mut self, w: &mut [f32], g: &[f32], eta: f32) {
        self.t += 1;
        let (b1, b2) = (0.9f32, 0.999f32);
        let (c1, c2) = (1.0 - b1.powi(self.t), 1.0 - b2.powi(self.t));
        for i in 0..w.len() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g[i] * g[i];
            w[i] -= eta * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + 1e-8);
        }
    }
}

/// The indices 0..n in a random order.
pub fn shuffled(n: usize, rng: &mut Rng) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.below(i + 1);
        v.swap(i, j);
    }
    v
}

/// Settings of one training run.
#[derive(Clone, Debug)]
pub struct Train {
    pub epochs: usize,
    pub batch: usize,
    pub eta: f32,
    pub decay: f32,
    /// Weight of each class in the loss (empty: all 1).
    pub class_weight: Vec<f32>,
    pub seed: u64,
}

/// Training and validation loss and accuracy after an epoch (epoch 0: before training).
#[derive(Clone, Debug)]
pub struct Point {
    pub epoch: usize,
    pub train: f32,
    pub val: f32,
    pub train_acc: f32,
    pub val_acc: f32,
}

/// Trains `net` with Adam in mini-batches; `augment(x, rng)` may change each training example
/// every time it is used (return it unchanged for no augmentation). Records the clean training
/// and validation loss after every epoch, and the loss of every mini-batch (before its step).
pub fn train(
    net: &mut Mlp,
    (xs, ys): (&[Vec<f32>], &[usize]),
    (xv, yv): (&[Vec<f32>], &[usize]),
    cfg: &Train,
    mut augment: impl FnMut(&[f32], &mut Rng) -> Vec<f32>,
) -> (Vec<Point>, Vec<f32>) {
    let mut rng = Rng::new(cfg.seed);
    let mut adam = Adam::new(net.params.len());
    let classes = *net.sizes.last().unwrap();
    let cw = if cfg.class_weight.is_empty() { vec![1.0; classes] } else { cfg.class_weight.clone() };
    let record = |net: &Mlp, epoch: usize| {
        let (train, train_acc) = net.loss_and_accuracy(xs, ys);
        let (val, val_acc) = net.loss_and_accuracy(xv, yv);
        Point { epoch, train, val, train_acc, val_acc }
    };
    let mut curve = vec![record(net, 0)];
    let mut steps = Vec::new();
    for epoch in 1..=cfg.epochs {
        let order = shuffled(xs.len(), &mut rng);
        for batch in order.chunks(cfg.batch.max(1)) {
            let bx: Vec<Vec<f32>> = batch.iter().map(|&n| augment(&xs[n], &mut rng)).collect();
            let by: Vec<usize> = batch.iter().map(|&n| ys[n]).collect();
            let idx: Vec<usize> = (0..batch.len()).collect();
            let (loss, g) = net.gradient(&bx, &by, &idx, &cw, cfg.decay);
            steps.push(loss);
            adam.step(&mut net.params, &g, cfg.eta);
        }
        curve.push(record(net, epoch));
    }
    (curve, steps)
}

/// The epoch with the lowest validation loss.
pub fn best(curve: &[Point]) -> &Point {
    curve.iter().fold(&curve[0], |b, p| if p.val < b.val { p } else { b })
}

// ---------------------------------------------------------------------------------------
// Data sets

/// A `size` × `size` crop with one smooth-edged ellipse: centre (cx, cy), semi-axes a (along
/// the direction `angle_deg`, clockwise from the x axis) and b, on a background of 0.1; the
/// inside has grey value `bright`. Each pixel is the average of 4 × 4 sub-pixel samples.
pub fn ellipse_crop(size: usize, (cx, cy): (f32, f32), (a, b): (f32, f32), angle_deg: f32, bright: f32) -> Image<f32> {
    let (s, c) = angle_deg.to_radians().sin_cos();
    Image::from_fn(size, size, |x, y| {
        let mut inside = 0.0;
        for sy in 0..4 {
            for sx in 0..4 {
                let dx = x as f32 + (sx as f32 + 0.5) / 4.0 - 0.5 - cx;
                let dy = y as f32 + (sy as f32 + 0.5) / 4.0 - 0.5 - cy;
                let (u, v) = (c * dx + s * dy, -s * dx + c * dy);
                if (u / a).powi(2) + (v / b).powi(2) <= 1.0 {
                    inside += 1.0 / 16.0;
                }
            }
        }
        0.1 + (bright - 0.1) * inside
    })
}

/// Which crop classification task.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Task {
    /// Class 0: round cells; class 1: elongated cells (in any direction).
    Shape,
    /// Class 0: lying (horizontal) elongated cells; class 1: standing (vertical) ones.
    Orientation,
}

/// `n` crops of 16 × 16 pixels, half of each class, with their labels. For `Shape`, the
/// training set (`train` = true) has its elongated cells all within 15° of horizontal and
/// brightness 0.8–1.0; the validation set has any direction and brightness 0.5–1.1. For
/// `Orientation`, both sets have cells within 15° of horizontal or vertical. Noise of standard
/// deviation 0.03 on every pixel.
pub fn crops(task: Task, n: usize, train: bool, seed: u64) -> (Vec<Image<f32>>, Vec<usize>) {
    let mut rng = Rng::new(seed);
    let mut imgs = Vec::new();
    let mut labels = Vec::new();
    for i in 0..n {
        let class = i % 2;
        let centre = (8.0 + 2.0 * rng.uniform() - 1.0, 8.0 + 2.0 * rng.uniform() - 1.0);
        let jitter = 30.0 * rng.uniform() - 15.0;
        let (axes, angle) = match (task, class) {
            (Task::Shape, 0) => {
                let r = 3.0 + 1.2 * rng.uniform();
                ((r, r * (0.9 + 0.1 * rng.uniform())), 180.0 * rng.uniform())
            }
            (Task::Shape, _) => {
                let angle = if train { jitter } else { 180.0 * rng.uniform() };
                ((5.0 + 1.5 * rng.uniform(), 2.0 + 0.7 * rng.uniform()), angle)
            }
            (Task::Orientation, c) => ((5.0 + 1.5 * rng.uniform(), 2.0 + 0.7 * rng.uniform()), 90.0 * c as f32 + jitter),
        };
        let bright = if train || task == Task::Orientation { 0.8 + 0.2 * rng.uniform() } else { 0.5 + 0.6 * rng.uniform() };
        let img = ellipse_crop(16, centre, axes, angle, bright);
        let img = img.map(|v| v + 0.03 * rng.normal());
        imgs.push(img);
        labels.push(class);
    }
    (imgs, labels)
}

/// The pixels of an image as one input vector, row by row.
pub fn to_input(img: &Image<f32>) -> Vec<f32> {
    img.pixels().to_vec()
}

/// A 16 × 16 image from an input vector.
pub fn from_input(x: &[f32]) -> Image<f32> {
    Image::from_vec(16, 16, x.to_vec())
}

/// Three-armed spiral: `counts[k]` points of class k, inputs in about [-1, 1]². Each arm turns
/// by about 200° from the centre outwards; `noise` is the standard deviation of the scatter.
pub fn spiral(counts: [usize; 3], noise: f32, seed: u64) -> (Vec<Vec<f32>>, Vec<usize>) {
    let mut rng = Rng::new(seed);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let total: usize = counts.iter().sum();
    let mut left = counts;
    // interleave the classes so that any prefix has them in proportion
    for i in 0..total {
        let k = (0..3).filter(|&k| left[k] > 0).max_by(|&a, &b| {
            let fa = left[a] as f32 / counts[a] as f32;
            let fb = left[b] as f32 / counts[b] as f32;
            fa.partial_cmp(&fb).unwrap().then(b.cmp(&a))
        });
        let k = k.unwrap_or(i % 3);
        left[k] -= 1;
        let t = 0.15 + 0.85 * rng.uniform();
        let theta = 2.0 * std::f32::consts::PI * k as f32 / 3.0 + 3.5 * t;
        xs.push(vec![t * theta.cos() + noise * rng.normal(), t * theta.sin() + noise * rng.normal()]);
        ys.push(k);
    }
    (xs, ys)
}

// ---------------------------------------------------------------------------------------
// Pictures

/// Line plot of several curves against their index (epoch or step), y from `y0` to `y1`
/// (values outside are clipped), 480 × 260 pixels, light grey lines at every `grid` in y.
pub fn plot_curves(curves: &[(&[f32], [f32; 3])], (y0, y1): (f32, f32), grid: f32, path: &str) {
    let (w, h) = (480usize, 260usize);
    let mut img = Image::new(w, h, [1.0f32; 3]);
    let n = curves.iter().map(|c| c.0.len()).max().unwrap_or(2).max(2);
    let row = |v: f32| ((h - 1) as f32 * (1.0 - ((v - y0) / (y1 - y0)).clamp(0.0, 1.0))).round() as isize;
    let mut g = y0;
    while g <= y1 + 1e-6 {
        let r = row(g) as usize;
        for x in 0..w {
            img[(x, r)] = [0.85; 3];
        }
        g += grid;
    }
    for (ys, colour) in curves {
        let col = |i: usize| (i as f32 * (w - 1) as f32 / (n - 1) as f32).round() as isize;
        for i in 1..ys.len() {
            if !ys[i].is_finite() || !ys[i - 1].is_finite() {
                continue;
            }
            let (xa, xb, ra, rb) = (col(i - 1), col(i), row(ys[i - 1]), row(ys[i]));
            let steps = (xb - xa).abs().max((rb - ra).abs()).max(1);
            for s in 0..=steps {
                let x = xa + (xb - xa) * s / steps;
                let y = ra + (rb - ra) * s / steps;
                for d in 0..2 {
                    if img.contains(x, y + d) {
                        img[(x as usize, (y + d) as usize)] = *colour;
                    }
                }
            }
        }
    }
    save_rgb(&img, path);
}

/// Colours for the plots.
pub const BLUE: [f32; 3] = [0.16, 0.44, 0.59];
pub const RED: [f32; 3] = [0.74, 0.28, 0.29];
pub const ORANGE: [f32; 3] = [0.88, 0.54, 0.12];
pub const GREEN: [f32; 3] = [0.23, 0.54, 0.36];
