// Helpers shared by the units of the day "GANs and normalising flows": the logistic function,
// one-dimensional data distributions, the complete one-dimensional GAN (the filled-in version of
// the unit gan's exercise, used by the program of gan-training), and a small line plot.

use imgcourse::*;

/// σ(s) = 1 / (1 + e^(−s)).
pub fn sigmoid(s: f64) -> f64 {
    1.0 / (1.0 + (-s).exp())
}

/// log σ(s), computed without overflow.
pub fn log_sigmoid(s: f64) -> f64 {
    -((-s).max(0.0) + (-s.abs()).exp().ln_1p())
}

/// The density of N(mu, sd²) at x.
pub fn normal_pdf(x: f64, mu: f64, sd: f64) -> f64 {
    (-0.5 * ((x - mu) / sd).powi(2)).exp() / (sd * (2.0 * std::f64::consts::PI).sqrt())
}

/// A standard normal number (from the library's f32 generator).
pub fn gauss(rng: &mut Rng) -> f64 {
    rng.normal() as f64
}

/// A mixture of Gaussians in 1D: weights, means and standard deviations.
#[derive(Clone, Debug)]
pub struct Mixture {
    pub w: Vec<f64>,
    pub mu: Vec<f64>,
    pub sd: Vec<f64>,
}

impl Mixture {
    /// One bump, N(1, 0.5²): the data of the unit gan.
    pub fn one_bump() -> Mixture {
        Mixture { w: vec![1.0], mu: vec![1.0], sd: vec![0.5] }
    }
    /// Two equal bumps at 0 and 2 with sd 0.3.
    pub fn two_bumps() -> Mixture {
        Mixture { w: vec![0.5, 0.5], mu: vec![0.0, 2.0], sd: vec![0.3, 0.3] }
    }
    pub fn pdf(&self, x: f64) -> f64 {
        (0..self.w.len()).map(|k| self.w[k] * normal_pdf(x, self.mu[k], self.sd[k])).sum()
    }
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        let mut u = rng.uniform() as f64;
        let mut k = 0;
        while k + 1 < self.w.len() && u > self.w[k] {
            u -= self.w[k];
            k += 1;
        }
        self.mu[k] + self.sd[k] * gauss(rng)
    }
}

/// Which loss the generator minimises: non-saturating −log D(G(z)), or the original
/// (minimax) log(1 − D(G(z))).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GenLoss {
    NonSaturating,
    Minimax,
}

/// The smallest GAN: generator x = a z + b (z ~ N(0, 1)), discriminator
/// D(x) = σ(c0 + c1 x + c2 x²).
#[derive(Clone, Debug)]
pub struct LineGan {
    pub a: f64,
    pub b: f64,
    pub c: [f64; 3],
}

/// Settings of one training run of a LineGan.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub lr_g: f64,
    pub lr_d: f64,
    /// Discriminator steps per generator step.
    pub k_d: usize,
    pub batch: usize,
    pub loss: GenLoss,
    /// Use the best discriminator D* = p/(p + q) (exact for a Gaussian generator) instead of the learned one.
    pub optimal: bool,
}

impl LineGan {
    /// The start of every run: the generator N(−1, 1), the discriminator D = 1/2 everywhere.
    pub fn start() -> LineGan {
        LineGan { a: 1.0, b: -1.0, c: [0.0; 3] }
    }

    /// The discriminator's logit at x (learned, or the best one for the data `data`).
    pub fn logit(&self, x: f64, data: &Mixture, optimal: bool) -> f64 {
        if optimal {
            let q = normal_pdf(x, self.b, self.a.abs());
            (data.pdf(x) / q).ln()
        } else {
            self.c[0] + self.c[1] * x + self.c[2] * x * x
        }
    }

    /// d logit / dx.
    fn slope(&self, x: f64, data: &Mixture, optimal: bool) -> f64 {
        if optimal {
            let h = 1e-5;
            (self.logit(x + h, data, true) - self.logit(x - h, data, true)) / (2.0 * h)
        } else {
            self.c[1] + 2.0 * self.c[2] * x
        }
    }

    /// One training step: k_d discriminator steps, then one generator step.
    pub fn train_step(&mut self, data: &Mixture, s: &Settings, rng: &mut Rng) {
        let n = s.batch as f64;
        if !s.optimal {
            for _ in 0..s.k_d {
                let mut g = [0.0; 3];
                for _ in 0..s.batch {
                    let xr = data.sample(rng);
                    let xf = self.a * gauss(rng) + self.b;
                    let dr = sigmoid(self.logit(xr, data, false)) - 1.0;
                    let df = sigmoid(self.logit(xf, data, false));
                    for (j, gj) in g.iter_mut().enumerate() {
                        *gj += dr * xr.powi(j as i32) + df * xf.powi(j as i32);
                    }
                }
                for j in 0..3 {
                    self.c[j] -= s.lr_d * g[j] / n;
                }
            }
        }
        let (mut ga, mut gb) = (0.0, 0.0);
        for _ in 0..s.batch {
            let z = gauss(rng);
            let x = self.a * z + self.b;
            let d = sigmoid(self.logit(x, data, s.optimal));
            let dl_ds = match s.loss {
                GenLoss::NonSaturating => -(1.0 - d),
                GenLoss::Minimax => -d,
            };
            let dx = dl_ds * self.slope(x, data, s.optimal);
            ga += dx * z;
            gb += dx;
        }
        self.a -= s.lr_g * ga / n;
        self.b -= s.lr_g * gb / n;
    }

    /// n samples of the generator.
    pub fn samples(&self, n: usize, rng: &mut Rng) -> Vec<f64> {
        (0..n).map(|_| self.a * gauss(rng) + self.b).collect()
    }
}

/// A curve of a plot: its (x, y) points and its colour.
pub type Curve = (Vec<(f64, f64)>, [f32; 3]);

/// A line plot: each curve is a list of (x, y) points and a colour; x in `xr`, y in `yr`.
pub fn plot(curves: &[Curve], xr: (f64, f64), yr: (f64, f64), w: usize, h: usize) -> Image<[f32; 3]> {
    let mut img = Image::new(w, h, [1.0f32; 3]);
    let px = |x: f64| ((x - xr.0) / (xr.1 - xr.0) * (w as f64 - 1.0)).round() as isize;
    let py = |y: f64| ((yr.1 - y) / (yr.1 - yr.0) * (h as f64 - 1.0)).round() as isize;
    // axes at x = 0 and y = 0 when visible
    for x in 0..w {
        let y0 = py(0.0);
        if img.contains(x as isize, y0) {
            img[(x, y0 as usize)] = [0.6; 3];
        }
    }
    for (pts, col) in curves {
        for win in pts.windows(2) {
            let (x0, y0, x1, y1) = (px(win[0].0), py(win[0].1), px(win[1].0), py(win[1].1));
            let steps = (x1 - x0).abs().max((y1 - y0).abs()).max(1);
            for k in 0..=steps {
                let x = x0 + (x1 - x0) * k / steps;
                let y = y0 + (y1 - y0) * k / steps;
                for dy in 0..2 {
                    if img.contains(x, y + dy) {
                        img[(x as usize, (y + dy) as usize)] = *col;
                    }
                }
            }
        }
    }
    img
}

pub const GREY: [f32; 3] = [0.55, 0.55, 0.55];
pub const BLUE: [f32; 3] = [0.12, 0.47, 0.71];
pub const RED: [f32; 3] = [0.84, 0.15, 0.16];
pub const GREEN: [f32; 3] = [0.17, 0.63, 0.17];
pub const ORANGE: [f32; 3] = [1.0, 0.5, 0.05];
