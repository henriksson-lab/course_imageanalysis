//! Given code for the unit "Few labels": a pool of cells in a 2D feature space and a small
//! probabilistic classifier (logistic regression on Gaussian bumps, fitted by Newton's method).

use imgcourse::linalg;
use imgcourse::Rng;

/// A cell: two measured features in [0, 1], its true class (0 or 1), and whether it belongs
/// to the rare subtype.
#[derive(Clone, Copy, Debug)]
pub struct Cell {
    pub x: f32,
    pub y: f32,
    pub class: usize,
    pub rare: bool,
}

/// `n` cells: two noisy moons (classes 0 and 1) and a small rare cluster of class 1 (6 %) far
/// from the boundary, in random order.
pub fn make_pool(n: usize, seed: u64) -> Vec<Cell> {
    let mut rng = Rng::new(seed);
    let n_rare = (0.06 * n as f32).round() as usize;
    let mut cells = Vec::new();
    for i in 0..n - n_rare {
        let class = i % 2;
        let t = std::f32::consts::PI * rng.uniform();
        let (x, y) = if class == 0 { (t.cos(), t.sin()) } else { (1.0 - t.cos(), 0.5 - t.sin()) };
        cells.push(Cell {
            x: 0.1 + 0.8 * (x + 1.0) / 3.0 + 0.06 * rng.normal(),
            y: 0.25 + 0.6 * (y + 0.5) / 1.5 + 0.06 * rng.normal(),
            class,
            rare: false,
        });
    }
    for _ in 0..n_rare {
        cells.push(Cell { x: 0.12 + 0.035 * rng.normal(), y: 0.88 + 0.035 * rng.normal(), class: 1, rare: true });
    }
    for i in (1..cells.len()).rev() {
        let j = rng.below(i + 1);
        cells.swap(i, j);
    }
    cells
}

/// Grid of Gaussian bumps: 7 × 7 centres, width 0.16; plus a constant.
const G: usize = 7;
const SIGMA: f64 = 0.16;
/// Penalty λ Σ w² (not on the constant).
const LAMBDA: f64 = 0.05;

fn feats(x: f32, y: f32) -> Vec<f64> {
    let mut f = vec![1.0];
    for i in 0..G {
        for j in 0..G {
            let (cx, cy) = ((i as f64 + 0.5) / G as f64, (j as f64 + 0.5) / G as f64);
            f.push((-((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)) / (2.0 * SIGMA * SIGMA)).exp());
        }
    }
    f
}

/// The classifier: p(class 1 | cell) = sigmoid(w · features).
#[derive(Clone, Debug)]
pub struct Model {
    pub w: Vec<f64>,
}

impl Model {
    /// Fitted to the labelled cells `idx` (penalised logistic regression, damped Newton steps).
    pub fn fit(cells: &[Cell], idx: &[usize]) -> Model {
        let nf = 1 + G * G;
        let phi: Vec<Vec<f64>> = idx.iter().map(|&i| feats(cells[i].x, cells[i].y)).collect();
        let ys: Vec<f64> = idx.iter().map(|&i| cells[i].class as f64).collect();
        let loss = |w: &[f64]| -> f64 {
            let mut l = 0.0;
            for (f, y) in phi.iter().zip(&ys) {
                let s: f64 = w.iter().zip(f).map(|(a, b)| a * b).sum();
                l += s.max(0.0) + (-s.abs()).exp().ln_1p() - y * s;
            }
            l + LAMBDA * w[1..].iter().map(|v| v * v).sum::<f64>()
        };
        let mut w = vec![0.0; nf];
        for _ in 0..30 {
            let mut g = vec![0.0; nf];
            let mut h = vec![vec![0.0; nf]; nf];
            for a in 1..nf {
                g[a] = 2.0 * LAMBDA * w[a];
                h[a][a] = 2.0 * LAMBDA;
            }
            h[0][0] += 1e-6;
            for (f, y) in phi.iter().zip(&ys) {
                let s: f64 = w.iter().zip(f).map(|(a, b)| a * b).sum();
                let p = 1.0 / (1.0 + (-s).exp());
                for a in 0..nf {
                    g[a] += (p - y) * f[a];
                    for b in 0..nf {
                        h[a][b] += p * (1.0 - p) * f[a] * f[b];
                    }
                }
            }
            let Some(d) = linalg::solve(&h, &g) else { break };
            let l0 = loss(&w);
            let mut t = 1.0;
            let mut wn = w.clone();
            for _ in 0..30 {
                wn = w.iter().zip(&d).map(|(a, b)| a - t * b).collect();
                if loss(&wn) <= l0 {
                    break;
                }
                t /= 2.0;
            }
            let step = w.iter().zip(&wn).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            w = wn;
            if step < 1e-8 {
                break;
            }
        }
        Model { w }
    }

    /// p(class 1) for a point.
    pub fn prob(&self, x: f32, y: f32) -> f32 {
        let s: f64 = self.w.iter().zip(feats(x, y)).map(|(a, b)| a * b).sum();
        (1.0 / (1.0 + (-s).exp())) as f32
    }

    /// Test accuracy, and the fraction of rare cells classified correctly.
    pub fn score(&self, test: &[Cell]) -> (f32, f32) {
        let ok = |c: &Cell| (self.prob(c.x, c.y) > 0.5) == (c.class == 1);
        let n_ok = test.iter().filter(|c| ok(c)).count();
        let rare: Vec<&Cell> = test.iter().filter(|c| c.rare).collect();
        let rare_ok = rare.iter().filter(|c| ok(c)).count();
        (n_ok as f32 / test.len() as f32, rare_ok as f32 / rare.len().max(1) as f32)
    }
}

/// Two cells of each class from the main moons (not the rare cluster), chosen by a seed.
pub fn start_set(cells: &[Cell], seed: u64) -> Vec<usize> {
    let mut rng = Rng::new(seed);
    let mut out: Vec<usize> = Vec::new();
    for class in [0, 1, 0, 1] {
        loop {
            let j = rng.below(cells.len());
            if cells[j].class == class && !cells[j].rare && !out.contains(&j) {
                out.push(j);
                break;
            }
        }
    }
    out
}
