//! Shared by the units of the day "Fine-tuning and transfer learning": the toy images
//! (10 × 10 patches with one object each), a small dense network with backpropagation, and its
//! pretraining on the "old task". These are given; the exercises are in the units' files.
//!
//! - Old task (lots of labels): four shapes, bright on dark: disc, ring, bar, two dots.
//! - New task (few labels): three cell phenotypes: interphase (a large dim nucleus with dark
//!   nucleoli), mitotic (a bright compact plate), apoptotic (bright fragments).
//! - A `Look` describes the microscope: pixel = 0.08 + offset + gain · object (+ noise).

use imgcourse::*;

/// Side of a patch in pixels; a patch is a `Vec<f32>` of `P * P` values, row by row.
pub const P: usize = 10;

pub const OLD_CLASSES: [&str; 4] = ["disc", "ring", "bar", "two dots"];
pub const NEW_CLASSES: [&str; 3] = ["interphase", "mitotic", "apoptotic"];

/// How the microscope renders an object.
#[derive(Clone, Copy, Debug)]
pub struct Look {
    pub gain: f32,
    pub offset: f32,
    pub noise: f32,
}

/// The training microscope (fluorescence).
pub const LOOK_A: Look = Look { gain: 1.0, offset: 0.0, noise: 0.06 };
/// Another modality: dark objects on a bright background.
pub const LOOK_FAR: Look = Look { gain: -0.6, offset: 0.82, noise: 0.06 };

fn sigmoid(t: f32) -> f32 {
    1.0 / (1.0 + (-t).exp())
}

/// One patch with an object of class `class` of the task (`old`: 4 shapes, else 3 phenotypes).
pub fn patch(old: bool, class: usize, rng: &mut Rng, look: Look) -> Vec<f32> {
    let kind = if old { OLD_CLASSES[class] } else { NEW_CLASSES[class] };
    let cx = 4.5 + (rng.uniform() - 0.5) * 2.0;
    let cy = 4.5 + (rng.uniform() - 0.5) * 2.0;
    let th = rng.uniform() * std::f32::consts::PI;
    let (c, s) = (th.cos(), th.sin());
    let a = 0.65 + 0.35 * rng.uniform();
    // spots: (x, y, sd)
    let mut dots: Vec<(f32, f32, f32)> = Vec::new();
    if kind == "two dots" {
        let h = 1.4 + 0.6 * rng.uniform();
        dots.push((cx + h * c, cy + h * s, 0.8));
        dots.push((cx - h * c, cy - h * s, 0.8));
    } else if kind == "apoptotic" {
        let n = 4 + rng.below(3);
        let a0 = rng.uniform() * 2.0 * std::f32::consts::PI;
        for i in 0..n {
            let ang = a0 + 2.0 * std::f32::consts::PI * (i as f32 + 0.3 * (rng.uniform() - 0.5)) / n as f32;
            let d = 1.6 + 1.4 * rng.uniform();
            dots.push((cx + d * ang.cos(), cy + d * ang.sin(), 0.5));
        }
    }
    let mut nucleoli: Vec<(f32, f32)> = Vec::new();
    if kind == "interphase" {
        for _ in 0..1 + rng.below(2) {
            let ang = rng.uniform() * 2.0 * std::f32::consts::PI;
            let d = 1.4 * rng.uniform().sqrt();
            nucleoli.push((cx + d * ang.cos(), cy + d * ang.sin()));
        }
    }
    let rad = match kind {
        "disc" => 2.0 + rng.uniform(),
        "ring" => 2.4 + 0.9 * rng.uniform(),
        _ => 2.8 + 0.8 * rng.uniform(),
    };
    let len = if kind == "bar" { 3.0 + 0.8 * rng.uniform() } else { 2.2 + 0.7 * rng.uniform() };
    let wid = if kind == "bar" { 0.9 + 0.3 * rng.uniform() } else { 0.65 + 0.2 * rng.uniform() };
    let mut out = vec![0.0; P * P];
    for y in 0..P {
        for x in 0..P {
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let d = (dx * dx + dy * dy).sqrt();
            let (u, w) = (dx * c + dy * s, -dx * s + dy * c);
            let mut v = match kind {
                "disc" => a * sigmoid((rad - d) / 0.45),
                "ring" => a * (-(d - rad) * (d - rad) / (2.0 * 0.55 * 0.55)).exp(),
                "bar" | "mitotic" => {
                    let e = ((u / len).powi(2) + (w / wid).powi(2)).sqrt();
                    (if kind == "mitotic" { 1.15 } else { 1.0 }) * a * sigmoid((1.0 - e) / 0.18)
                }
                "interphase" => {
                    let mut v = 0.55 * a * sigmoid((rad - d) / 0.5);
                    for &(px, py) in &nucleoli {
                        let r2 = (x as f32 - px).powi(2) + (y as f32 - py).powi(2);
                        v *= 1.0 - 0.75 * (-r2 / (2.0 * 0.36)).exp();
                    }
                    v
                }
                _ => 0.0,
            };
            for &(px, py, sd) in &dots {
                v += a * (-((x as f32 - px).powi(2) + (y as f32 - py).powi(2)) / (2.0 * sd * sd)).exp();
            }
            out[y * P + x] = 0.08 + look.offset + look.gain * v + look.noise * rng.normal();
        }
    }
    out
}

/// A labelled set of `n` patches, classes in turn (balanced). The same seed with another look
/// gives the same objects seen through that microscope.
pub fn make_set(old: bool, n: usize, seed: u64, look: Look) -> (Vec<Vec<f32>>, Vec<usize>) {
    let k = if old { 4 } else { 3 };
    let mut rng = Rng::new(seed);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for i in 0..n {
        xs.push(patch(old, i % k, &mut rng, look));
        ys.push(i % k);
    }
    (xs, ys)
}

// ---------------------------------------------------------------------------------------------
// A dense network: ReLU hidden layers, softmax + cross-entropy at the end.

/// One layer: `out = W · in + b`, with `w[o][i]` the weight from input `i` to output `o`.
#[derive(Clone, Debug)]
pub struct Layer {
    pub w: Vec<Vec<f32>>,
    pub b: Vec<f32>,
}

impl Layer {
    /// He initialisation (standard deviation sqrt(2 / inputs)), zero biases.
    pub fn random(n_in: usize, n_out: usize, rng: &mut Rng) -> Layer {
        let s = (2.0 / n_in as f32).sqrt();
        Layer { w: (0..n_out).map(|_| (0..n_in).map(|_| s * rng.normal()).collect()).collect(), b: vec![0.0; n_out] }
    }
    pub fn zeros_like(&self) -> Layer {
        Layer { w: self.w.iter().map(|r| vec![0.0; r.len()]).collect(), b: vec![0.0; self.b.len()] }
    }
    /// Number of weights and biases.
    pub fn count(&self) -> usize {
        self.w.len() * self.w[0].len() + self.b.len()
    }
}

/// A network is a list of layers; ReLU after every layer but the last.
pub type Net = Vec<Layer>;

/// A fresh network with the given sizes, e.g. `[100, 32, 16, 3]`.
pub fn new_net(sizes: &[usize], seed: u64) -> Net {
    let mut rng = Rng::new(seed);
    sizes.windows(2).map(|s| Layer::random(s[0], s[1], &mut rng)).collect()
}

/// The outputs of every layer for one input: `a[0]` = the input, `a[l + 1]` = after layer `l`
/// (after its ReLU, except the last, which holds the class scores).
pub fn forward_all(net: &Net, x: &[f32]) -> Vec<Vec<f32>> {
    let mut a = vec![x.to_vec()];
    for (l, layer) in net.iter().enumerate() {
        let last = l + 1 == net.len();
        let inp = &a[l];
        let out: Vec<f32> = layer
            .w
            .iter()
            .zip(&layer.b)
            .map(|(row, b)| {
                let s = b + row.iter().zip(inp).map(|(w, v)| w * v).sum::<f32>();
                if last || s > 0.0 {
                    s
                } else {
                    0.0
                }
            })
            .collect();
        a.push(out);
    }
    a
}

/// Softmax probabilities of a score vector.
pub fn softmax(z: &[f32]) -> Vec<f32> {
    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

/// The class with the highest score.
pub fn predict(net: &Net, x: &[f32]) -> usize {
    let a = forward_all(net, x);
    argmax(a.last().unwrap())
}

pub fn argmax(v: &[f32]) -> usize {
    (0..v.len()).fold(0, |b, j| if v[j] > v[b] { j } else { b })
}

/// Fraction of examples classified correctly.
pub fn net_accuracy(net: &Net, xs: &[Vec<f32>], ys: &[usize]) -> f32 {
    xs.iter().zip(ys).filter(|(x, &y)| predict(net, x) == y).count() as f32 / xs.len() as f32
}

/// Gradients of the mean cross-entropy over the examples `idx` with respect to every weight
/// (backpropagation), and the mean loss.
pub fn gradients(net: &Net, xs: &[Vec<f32>], ys: &[usize], idx: &[usize]) -> (Net, f32) {
    let mut g: Net = net.iter().map(|l| l.zeros_like()).collect();
    let mut loss = 0.0;
    let m = idx.len() as f32;
    for &i in idx {
        let a = forward_all(net, &xs[i]);
        let p = softmax(a.last().unwrap());
        loss -= p[ys[i]].max(1e-30).ln();
        // δ at the scores: p − one-hot
        let mut d: Vec<f32> = p.iter().enumerate().map(|(j, &pj)| (pj - if j == ys[i] { 1.0 } else { 0.0 }) / m).collect();
        for l in (0..net.len()).rev() {
            let inp = &a[l];
            let mut d_prev = vec![0.0; inp.len()];
            for o in 0..d.len() {
                g[l].b[o] += d[o];
                for k in 0..inp.len() {
                    g[l].w[o][k] += d[o] * inp[k];
                    d_prev[k] += d[o] * net[l].w[o][k];
                }
            }
            // through the ReLU of the layer below (its output is inp): blocked where it was 0
            if l > 0 {
                for k in 0..inp.len() {
                    if inp[k] <= 0.0 {
                        d_prev[k] = 0.0;
                    }
                }
            }
            d = d_prev;
        }
    }
    (g, loss / m)
}

/// Plain gradient descent: every weight of layer `l` moves by `-rates[l] * gradient`.
pub fn descend(net: &mut Net, g: &Net, rates: &[f32]) {
    for (l, layer) in net.iter_mut().enumerate() {
        for (row, grow) in layer.w.iter_mut().zip(&g[l].w) {
            for (w, gw) in row.iter_mut().zip(grow) {
                *w -= rates[l] * gw;
            }
        }
        for (b, gb) in layer.b.iter_mut().zip(&g[l].b) {
            *b -= rates[l] * gb;
        }
    }
}

/// `steps` steps of mini-batch gradient descent (batches of `batch`, 0 = all), with one learning
/// rate per layer (0 freezes a layer). Returns the last batch loss.
pub fn train(net: &mut Net, xs: &[Vec<f32>], ys: &[usize], rates: &[f32], steps: usize, batch: usize, seed: u64) -> f32 {
    let mut rng = Rng::new(seed);
    let mut loss = 0.0;
    let all: Vec<usize> = (0..xs.len()).collect();
    for _ in 0..steps {
        let idx: Vec<usize> = if batch == 0 || batch >= xs.len() { all.clone() } else { (0..batch).map(|_| rng.below(xs.len())).collect() };
        let (g, l) = gradients(net, xs, ys, &idx);
        descend(net, &g, rates);
        loss = l;
    }
    loss
}

/// The network of the old task: 100 → 32 → 16 → 4, trained on 2400 labelled patches
/// (3000 steps of batches of 32, learning rate 0.05). Takes a few seconds.
pub fn pretrained() -> Net {
    let (xs, ys) = make_set(true, 2400, 101, LOOK_A);
    let mut net = new_net(&[P * P, 32, 16, 4], 7);
    train(&mut net, &xs, &ys, &[0.05, 0.05, 0.05], 3000, 32, 102);
    net
}

/// The pretrained backbone with a new random head for `k` classes.
pub fn with_new_head(pre: &Net, k: usize, seed: u64) -> Net {
    let mut net = pre.clone();
    let n_in = net[net.len() - 1].w[0].len();
    let mut rng = Rng::new(seed);
    let last = net.len() - 1;
    net[last] = Layer::random(n_in, k, &mut rng);
    net
}

/// A first-layer weight row (a filter on a patch) as a picture, scaled to [0, 1] around 0.5.
pub fn filter_image(row: &[f32]) -> Image<f32> {
    let m = row.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    Image::from_fn(P, P, |x, y| 0.5 + 0.5 * row[y * P + x] / m)
}

/// Several P × P pictures side by side (with a 1-pixel gap), each enlarged `zoom` times.
pub fn mosaic(tiles: &[Image<f32>], per_row: usize, zoom: usize) -> Image<f32> {
    let rows = tiles.len().div_ceil(per_row);
    let s = P * zoom + 1;
    let mut out = Image::new(per_row * s, rows * s, 1.0);
    for (t, tile) in tiles.iter().enumerate() {
        let (ox, oy) = ((t % per_row) * s, (t / per_row) * s);
        for y in 0..P * zoom {
            for x in 0..P * zoom {
                out[(ox + x, oy + y)] = tile[(x / zoom, y / zoom)].clamp(0.0, 1.0);
            }
        }
    }
    out
}
