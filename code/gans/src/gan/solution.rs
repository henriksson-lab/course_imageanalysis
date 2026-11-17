// Model solution for the coding exercises of "Generative adversarial networks".

pub use crate::common::{sigmoid, GenLoss, LineGan, Mixture, Settings};
use crate::common::{gauss, log_sigmoid};
use imgcourse::Rng;

/// C1. The discriminator D(x) = σ(c0 + c1 x + c2 x²): the probability that x is real.
pub fn discriminator(c: &[f64; 3], x: f64) -> f64 {
    sigmoid(c[0] + c[1] * x + c[2] * x * x)
}

/// C2. The best discriminator for data density `p_data` and generator density `p_gen` at a point:
/// D*(x) = p_data / (p_data + p_gen) (one half where both are zero).
pub fn optimal_discriminator(p_data: f64, p_gen: f64) -> f64 {
    if p_data + p_gen > 0.0 {
        p_data / (p_data + p_gen)
    } else {
        0.5
    }
}

/// C3. The discriminator's loss: the mean of −log D(x) over the real points plus the mean of
/// −log(1 − D(x)) over the generated points.
pub fn loss_d(c: &[f64; 3], real: &[f64], fake: &[f64]) -> f64 {
    let s = |x: f64| c[0] + c[1] * x + c[2] * x * x;
    let lr: f64 = real.iter().map(|&x| -log_sigmoid(s(x))).sum::<f64>() / real.len() as f64;
    let lf: f64 = fake.iter().map(|&x| -log_sigmoid(-s(x))).sum::<f64>() / fake.len() as f64;
    lr + lf
}

/// C3. The gradient of `loss_d` with respect to (c0, c1, c2).
pub fn grad_d(c: &[f64; 3], real: &[f64], fake: &[f64]) -> [f64; 3] {
    let mut g = [0.0; 3];
    for &x in real {
        let d = discriminator(c, x) - 1.0; // ∂(−log σ(s))/∂s = σ(s) − 1
        g[0] += d / real.len() as f64;
        g[1] += d * x / real.len() as f64;
        g[2] += d * x * x / real.len() as f64;
    }
    for &x in fake {
        let d = discriminator(c, x); // ∂(−log(1 − σ(s)))/∂s = σ(s)
        g[0] += d / fake.len() as f64;
        g[1] += d * x / fake.len() as f64;
        g[2] += d * x * x / fake.len() as f64;
    }
    g
}

/// C4. The gradient of the generator's loss (mean over the noise values `zs`) with respect to
/// (a, b), for the generator x = a z + b and the discriminator with weights `gan.c`.
pub fn grad_g(gan: &LineGan, zs: &[f64], loss: GenLoss) -> (f64, f64) {
    let (mut ga, mut gb) = (0.0, 0.0);
    for &z in zs {
        let x = gan.a * z + gan.b;
        let d = discriminator(&gan.c, x);
        let dl_ds = match loss {
            GenLoss::NonSaturating => -(1.0 - d),
            GenLoss::Minimax => -d,
        };
        let dx = dl_ds * (gan.c[1] + 2.0 * gan.c[2] * x); // chain rule: ∂s/∂x
        ga += dx * z / zs.len() as f64;
        gb += dx / zs.len() as f64;
    }
    (ga, gb)
}

/// C5. One training step: `k_d` discriminator steps (each with a fresh batch of real points
/// from `data` and generated points), then one generator step with fresh noise. Plain gradient
/// descent with the learning rates of `s` (`s.optimal` is not used here).
pub fn train_step(gan: &mut LineGan, data: &Mixture, s: &Settings, rng: &mut Rng) {
    for _ in 0..s.k_d {
        let real: Vec<f64> = (0..s.batch).map(|_| data.sample(rng)).collect();
        let fake: Vec<f64> = (0..s.batch).map(|_| gan.a * gauss(rng) + gan.b).collect();
        let g = grad_d(&gan.c, &real, &fake);
        for j in 0..3 {
            gan.c[j] -= s.lr_d * g[j];
        }
    }
    let zs: Vec<f64> = (0..s.batch).map(|_| gauss(rng)).collect();
    let (ga, gb) = grad_g(gan, &zs, s.loss);
    gan.a -= s.lr_g * ga;
    gan.b -= s.lr_g * gb;
}
