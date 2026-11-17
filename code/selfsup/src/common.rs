//! Helpers shared by the units of the day "Learning without labels" (not exercises).

use imgcourse::{filters, Border, Image, Rng};

/// Pure noise of standard deviation `sigma`, independent from pixel to pixel
/// (`corr = 0`), or blurred with a Gaussian of `corr` pixels and scaled back to `sigma`:
/// noise that is correlated between neighbours, as after demosaicing or resampling.
pub fn noise(width: usize, height: usize, sigma: f32, corr: f32, seed: u64) -> Image<f32> {
    let mut rng = Rng::new(seed);
    let white = Image::from_fn(width, height, |_, _| rng.normal());
    if corr <= 0.0 {
        return white.map(|v| sigma * v);
    }
    let blurred = filters::gaussian_blur(&white, corr);
    let n = blurred.pixels().len() as f32;
    let sd = (blurred.pixels().iter().map(|v| v * v).sum::<f32>() / n).sqrt();
    blurred.map(|v| sigma * v / sd)
}

/// Non-local means (the deconvolution day's denoising unit, C3, filled in): a weighted mean
/// over the search window `(2 search + 1)²` (inside the image), weights `exp(−d² / h²)` with
/// `d²` the mean squared difference of the `(2 patch + 1)²` patches (mirror border).
pub fn nonlocal_means(img: &Image<f32>, patch: usize, search: usize, h: f32) -> Image<f32> {
    let (pr, sr) = (patch as isize, search as isize);
    let read = |x: isize, y: isize| img.get_border(x, y, Border::Mirror);
    let patch_size = ((2 * pr + 1) * (2 * pr + 1)) as f32;
    let mut out = Image::new(img.width(), img.height(), 0.0);
    for (x, y) in img.coords() {
        let (px, py) = (x as isize, y as isize);
        let (mut sum, mut total) = (0.0, 0.0);
        for qy in py - sr..=py + sr {
            for qx in px - sr..=px + sr {
                if !img.contains(qx, qy) {
                    continue;
                }
                let mut d2 = 0.0;
                for dy in -pr..=pr {
                    for dx in -pr..=pr {
                        d2 += (read(px + dx, py + dy) - read(qx + dx, qy + dy)).powi(2);
                    }
                }
                let w = (-(d2 / patch_size) / (h * h)).exp();
                sum += w * img[(qx as usize, qy as usize)];
                total += w;
            }
        }
        out[(x, y)] = sum / total;
    }
    out
}

/// An image enlarged `k` times by repeating pixels (for looking at small pictures).
pub fn enlarge(img: &Image<f32>, k: usize) -> Image<f32> {
    Image::from_fn(img.width() * k, img.height() * k, |x, y| img[(x / k, y / k)])
}

/// Images side by side with a white gap of 2 pixels.
pub fn side_by_side(imgs: &[&Image<f32>]) -> Image<f32> {
    let h = imgs.iter().map(|i| i.height()).max().unwrap_or(0);
    let w: usize = imgs.iter().map(|i| i.width() + 2).sum::<usize>().saturating_sub(2);
    let mut out = Image::new(w, h, 1.0);
    let mut x0 = 0;
    for img in imgs {
        for (x, y) in img.coords() {
            out[(x0 + x, y)] = img[(x, y)];
        }
        x0 += img.width() + 2;
    }
    out
}

// ---------------------------------------------------------------------------------------
// The toy of the contrastive and self-distillation units: small synthetic "cells" and a tiny
// encoder (the same toy as the demo of the contrastive unit, with its own random numbers).

/// Side of a cell picture: 16 × 16 pixels.
pub const CELL: usize = 16;

/// A synthetic cell: its picture (row by row) and two labels that training never sees.
#[derive(Clone, Debug)]
pub struct Cell {
    pub pixels: Vec<f32>,
    /// 0 round, 1 elongated, 2 a dividing pair.
    pub shape: usize,
    /// staining brightness, 0.35 to 1 (dim below 0.675).
    pub bright: f32,
}

fn render(shape: usize, angle: f32, b: f32, ox: f32, oy: f32) -> Vec<f32> {
    let (c, s) = (angle.cos(), angle.sin());
    let soft = |d: f32| 1.0 / (1.0 + (d / 0.35).exp());
    let mut v = vec![0.0; CELL * CELL];
    for y in 0..CELL {
        for x in 0..CELL {
            let (px, py) = (x as f32 + 0.5 - 8.0 - ox, y as f32 + 0.5 - 8.0 - oy);
            let (u, w) = (c * px + s * py, -s * px + c * py);
            let m = match shape {
                0 => soft(u.hypot(w) - 3.6),
                1 => soft(((u / 5.2).hypot(w / 2.1) - 1.0) * 2.1),
                _ => soft((u - 3.1).hypot(w) - 2.4).max(soft((u + 3.1).hypot(w) - 2.4)),
            };
            v[y * CELL + x] = b * m;
        }
    }
    v
}

/// `n` cells, shapes in turn, random orientation, brightness and small offset.
pub fn make_cells(n: usize, seed: u64) -> Vec<Cell> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|i| {
            let shape = i % 3;
            let bright = 0.35 + 0.65 * rng.uniform();
            let angle = std::f32::consts::PI * rng.uniform();
            let (ox, oy) = (rng.uniform() - 0.5, rng.uniform() - 0.5);
            Cell { pixels: render(shape, angle, bright, ox, oy), shape, bright }
        })
        .collect()
}

/// Which augmentations to use, and how strong (0 = none, 1 = full).
#[derive(Clone, Copy, Debug)]
pub struct Augment {
    pub strength: f32,
    pub geometry: bool,
    pub brightness: bool,
    pub noise: bool,
}

/// One random view: rotation (up to ±180° · strength) and shift (±2 px) about the centre,
/// bilinear, outside = 0; brightness gain `exp(±1.2 · strength)`; noise sd `0.05 · strength`.
pub fn augment(pixels: &[f32], a: Augment, rng: &mut Rng) -> Vec<f32> {
    let t = a.strength;
    let mut sym = |on: bool, k: f32| if on { k * t * (2.0 * rng.uniform() - 1.0) } else { 0.0 };
    let angle = sym(a.geometry, std::f32::consts::PI);
    let (dx, dy) = (sym(a.geometry, 2.0), sym(a.geometry, 2.0));
    let gain = sym(a.brightness, 1.2).exp();
    let (c, s) = (angle.cos(), angle.sin());
    let at = |x: isize, y: isize| if x < 0 || y < 0 || x >= CELL as isize || y >= CELL as isize { 0.0 } else { pixels[y as usize * CELL + x as usize] };
    let mut out = vec![0.0; CELL * CELL];
    for y in 0..CELL {
        for x in 0..CELL {
            let (px, py) = (x as f32 + 0.5 - 8.0 - dx, y as f32 + 0.5 - 8.0 - dy);
            let (sx, sy) = (c * px - s * py + 7.5, s * px + c * py + 7.5);
            let (x0, y0) = (sx.floor(), sy.floor());
            let (fx, fy) = (sx - x0, sy - y0);
            let (x0, y0) = (x0 as isize, y0 as isize);
            let v = (1.0 - fx) * (1.0 - fy) * at(x0, y0) + fx * (1.0 - fy) * at(x0 + 1, y0) + (1.0 - fx) * fy * at(x0, y0 + 1) + fx * fy * at(x0 + 1, y0 + 1);
            out[y * CELL + x] = gain * v + if a.noise { 0.05 * t * rng.normal() } else { 0.0 };
        }
    }
    out
}

/// A tiny encoder: `out = W2 relu(W1 x + b1) + b2` (the backpropagation unit's network, wider).
#[derive(Clone, Debug)]
pub struct Mlp {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub n_in: usize,
    pub n_hidden: usize,
    pub n_out: usize,
}

/// The values of one forward pass (kept for the backward pass).
pub struct Pass {
    pub x: Vec<f32>,
    pub h: Vec<f32>,
    pub out: Vec<f32>,
}

impl Mlp {
    pub fn new(n_in: usize, n_hidden: usize, n_out: usize, seed: u64) -> Mlp {
        let mut rng = Rng::new(seed);
        let s1 = (2.0 / n_in as f32).sqrt();
        let s2 = (1.0 / n_hidden as f32).sqrt();
        Mlp {
            w1: (0..n_in * n_hidden).map(|_| s1 * rng.normal()).collect(),
            b1: vec![0.0; n_hidden],
            w2: (0..n_hidden * n_out).map(|_| s2 * rng.normal()).collect(),
            b2: vec![0.0; n_out],
            n_in,
            n_hidden,
            n_out,
        }
    }

    /// The same shape with all weights 0 (for gradients).
    pub fn zeros_like(&self) -> Mlp {
        Mlp { w1: vec![0.0; self.w1.len()], b1: vec![0.0; self.b1.len()], w2: vec![0.0; self.w2.len()], b2: vec![0.0; self.b2.len()], ..*self }
    }

    pub fn forward(&self, x: &[f32]) -> Pass {
        let h: Vec<f32> = (0..self.n_hidden)
            .map(|j| (self.b1[j] + (0..self.n_in).map(|i| self.w1[j * self.n_in + i] * x[i]).sum::<f32>()).max(0.0))
            .collect();
        let out = (0..self.n_out).map(|k| self.b2[k] + (0..self.n_hidden).map(|j| self.w2[k * self.n_hidden + j] * h[j]).sum::<f32>()).collect();
        Pass { x: x.to_vec(), h, out }
    }

    /// Add the gradient for `d_out` = dLoss/d(out) of one pass to `g`.
    pub fn backward(&self, p: &Pass, d_out: &[f32], g: &mut Mlp) {
        let mut dh = vec![0.0; self.n_hidden];
        for k in 0..self.n_out {
            g.b2[k] += d_out[k];
            for j in 0..self.n_hidden {
                g.w2[k * self.n_hidden + j] += d_out[k] * p.h[j];
                dh[j] += d_out[k] * self.w2[k * self.n_hidden + j];
            }
        }
        for j in 0..self.n_hidden {
            if p.h[j] <= 0.0 {
                continue; // the ReLU blocks
            }
            g.b1[j] += dh[j];
            for i in 0..self.n_in {
                g.w1[j * self.n_in + i] += dh[j] * p.x[i];
            }
        }
    }

    fn params_mut(&mut self) -> [&mut Vec<f32>; 4] {
        [&mut self.w1, &mut self.b1, &mut self.w2, &mut self.b2]
    }
}

/// Adam (the training day): per-weight steps from running averages of g and g².
pub struct Adam {
    m: Mlp,
    v: Mlp,
    t: i32,
    pub lr: f32,
}

impl Adam {
    pub fn new(net: &Mlp, lr: f32) -> Adam {
        Adam { m: net.zeros_like(), v: net.zeros_like(), t: 0, lr }
    }

    pub fn step(&mut self, net: &mut Mlp, g: &mut Mlp) {
        self.t += 1;
        let (b1, b2) = (0.9f32, 0.999f32);
        let (c1, c2) = (1.0 - b1.powi(self.t), 1.0 - b2.powi(self.t));
        let lr = self.lr;
        for ((p, gr), (m, v)) in net.params_mut().into_iter().zip(g.params_mut()).zip(self.m.params_mut().into_iter().zip(self.v.params_mut())) {
            for i in 0..p.len() {
                m[i] = b1 * m[i] + (1.0 - b1) * gr[i];
                v[i] = b2 * v[i] + (1.0 - b2) * gr[i] * gr[i];
                p[i] -= lr * (m[i] / c1) / ((v[i] / c2).sqrt() + 1e-8);
            }
        }
    }
}

/// Leave-one-out k-nearest-neighbour accuracy of `labels` (with `classes` classes) on the
/// feature vectors `f` (squared Euclidean distance; ties go to the lower class).
pub fn knn_accuracy(f: &[Vec<f32>], labels: &[usize], classes: usize, k: usize) -> f32 {
    let mut ok = 0;
    for i in 0..f.len() {
        let mut d: Vec<(f32, usize)> = (0..f.len())
            .filter(|&j| j != i)
            .map(|j| (f[i].iter().zip(&f[j]).map(|(a, b)| (a - b) * (a - b)).sum::<f32>(), j))
            .collect();
        d.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut votes = vec![0; classes];
        for &(_, j) in d.iter().take(k) {
            votes[labels[j]] += 1;
        }
        let best = (0..classes).max_by_key(|&c| (votes[c], std::cmp::Reverse(c))).unwrap_or(0);
        if best == labels[i] {
            ok += 1;
        }
    }
    ok as f32 / f.len() as f32
}

/// The vector scaled to length 1.
pub fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / n).collect()
}
