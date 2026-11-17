//! Given code for the unit "Domain shift": cells measured by two features (mean intensity and
//! size) on two microscopes, and a linear classifier (logistic regression by Newton's method).

use imgcourse::linalg;
use imgcourse::Rng;

/// A cell: features `f = [mean intensity, size]` and its class (0 interphase, 1 mitotic).
#[derive(Clone, Copy, Debug)]
pub struct Cell {
    pub f: [f32; 2],
    pub class: usize,
}

/// `n` cells measured by microscope A; a share `frac_mitotic` of them mitotic (brighter, smaller).
pub fn cells_a(n: usize, seed: u64, frac_mitotic: f32) -> Vec<Cell> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let class = (rng.uniform() < frac_mitotic) as usize;
            let i = if class == 1 { 0.62 } else { 0.42 } + 0.08 * rng.normal();
            let s = if class == 1 { 0.40 } else { 0.58 } + 0.09 * rng.normal();
            Cell { f: [i, s], class }
        })
        .collect()
}

/// The same cells measured by microscope B at shift `s` in [0, 1]: intensity × (1 − 0.55 s) + 0.12 s.
pub fn to_b(cells: &[Cell], s: f32) -> Vec<Cell> {
    cells.iter().map(|c| Cell { f: [(1.0 - 0.55 * s) * c.f[0] + 0.12 * s, c.f[1]], class: c.class }).collect()
}

/// A linear classifier: p(mitotic) = sigmoid(w0 + w1 f0 + w2 f1).
#[derive(Clone, Copy, Debug)]
pub struct Linear {
    pub w: [f64; 3],
}

impl Linear {
    /// Logistic regression with the penalty λ ‖w − prior‖², by Newton's method. With `prior`
    /// the weights of another classifier and a larger λ, this fine-tunes that classifier.
    pub fn fit(cells: &[Cell], lambda: f64, prior: [f64; 3]) -> Linear {
        let mut w = prior;
        for _ in 0..30 {
            let mut g = vec![0.0; 3];
            let mut h = vec![vec![0.0; 3]; 3];
            for c in cells {
                let x = [1.0, c.f[0] as f64, c.f[1] as f64];
                let s = w[0] + w[1] * x[1] + w[2] * x[2];
                let p = 1.0 / (1.0 + (-s).exp());
                for a in 0..3 {
                    g[a] += (p - c.class as f64) * x[a];
                    for b in 0..3 {
                        h[a][b] += p * (1.0 - p) * x[a] * x[b];
                    }
                }
            }
            for a in 0..3 {
                g[a] += 2.0 * lambda * (w[a] - prior[a]);
                h[a][a] += 2.0 * lambda;
            }
            let Some(d) = linalg::solve(&h, &g) else { break };
            for a in 0..3 {
                w[a] -= d[a];
            }
            if d.iter().all(|v| v.abs() < 1e-10) {
                break;
            }
        }
        Linear { w }
    }

    pub fn accuracy(&self, cells: &[Cell]) -> f32 {
        let ok = cells
            .iter()
            .filter(|c| {
                let s = self.w[0] + self.w[1] * c.f[0] as f64 + self.w[2] * c.f[1] as f64;
                (s > 0.0) == (c.class == 1)
            })
            .count();
        ok as f32 / cells.len() as f32
    }
}
