// Model solution for the coding exercises of "Bayesian image denoising with an Ising prior".

use imgcourse::{Image, Rng};

/// A binary image: label 1 = object, 0 = background.
pub type Labels = Image<u32>;

/// The data costs of every pixel, `[D_p(0), D_p(1)]`: what label 0 and label 1 cost there.
pub type Costs = Image<[f32; 2]>;

/// Data costs for flip noise: 0 for the observed label, `ln((1 − ε)/ε)` for the other one.
pub fn flip_noise_costs(y: &Labels, eps: f32) -> Costs {
    let c = ((1.0 - eps) / eps).ln();
    y.map(|v| if v == 1 { [c, 0.0] } else { [0.0, c] })
}

/// The 4-neighbours of `(x, y)` inside a `width × height` image.
pub fn neighbours(x: usize, y: usize, width: usize, height: usize) -> Vec<(usize, usize)> {
    let mut n = Vec::with_capacity(4);
    if x > 0 {
        n.push((x - 1, y));
    }
    if x + 1 < width {
        n.push((x + 1, y));
    }
    if y > 0 {
        n.push((x, y - 1));
    }
    if y + 1 < height {
        n.push((x, y + 1));
    }
    n
}

/// C1. The posterior energy `E(L) = Σ_p D_p(L_p) + λ · (number of 4-neighbour pairs with
/// different labels)`, each pair counted once.
pub fn energy(labels: &Labels, costs: &Costs, lambda: f32) -> f32 {
    let (w, h) = (labels.width(), labels.height());
    let mut e = 0.0;
    for (x, y) in labels.coords() {
        e += costs[(x, y)][labels[(x, y)] as usize];
        // right and lower neighbour: every pair once
        if x + 1 < w && labels[(x + 1, y)] != labels[(x, y)] {
            e += lambda;
        }
        if y + 1 < h && labels[(x, y + 1)] != labels[(x, y)] {
            e += lambda;
        }
    }
    e
}

/// The energy of giving pixel `(x, y)` the label `l`, with its neighbours as they are:
/// `D_p(l) + λ · (number of neighbours with a different label)`.
pub fn local_energy(labels: &Labels, costs: &Costs, lambda: f32, x: usize, y: usize, l: u32) -> f32 {
    let differ = neighbours(x, y, labels.width(), labels.height()).iter().filter(|&&q| labels[q] != l).count();
    costs[(x, y)][l as usize] + lambda * differ as f32
}

/// C2. Iterated conditional modes: visit the pixels in the given `order` and give each the
/// label with the lower local energy (keep the label on a tie); repeat until a whole sweep
/// changes nothing. Returns the result and the number of sweeps (the last one changes nothing).
pub fn icm(start: &Labels, costs: &Costs, lambda: f32, order: &[(usize, usize)]) -> (Labels, usize) {
    let mut labels = start.clone();
    let mut sweeps = 0;
    loop {
        sweeps += 1;
        let mut changed = false;
        for &(x, y) in order {
            let old = labels[(x, y)];
            let other = 1 - old;
            if local_energy(&labels, costs, lambda, x, y, other) < local_energy(&labels, costs, lambda, x, y, old) {
                labels[(x, y)] = other;
                changed = true;
            }
        }
        if !changed {
            return (labels, sweeps);
        }
    }
}

/// C3. One Gibbs sweep at temperature `t`, row by row: draw each label from its local
/// conditional, label 1 with probability `1 / (1 + e^(Δ/T))`, `Δ = E(L_p = 1) − E(L_p = 0)`
/// with the neighbours as they are.
pub fn gibbs_sweep(labels: &mut Labels, costs: &Costs, lambda: f32, t: f32, rng: &mut Rng) {
    for y in 0..labels.height() {
        for x in 0..labels.width() {
            let delta = local_energy(labels, costs, lambda, x, y, 1) - local_energy(labels, costs, lambda, x, y, 0);
            let p1 = 1.0 / (1.0 + (delta / t).exp());
            labels[(x, y)] = if rng.uniform() < p1 { 1 } else { 0 };
        }
    }
}

/// C4. The marginals `p(L_p = 1 | y)` by sampling at `T = 1`: start from `start`, do
/// `burn_in` Gibbs sweeps, then `sweeps` more, and return for every pixel the fraction of
/// those `sweeps` in which it was 1.
pub fn marginals(start: &Labels, costs: &Costs, lambda: f32, burn_in: usize, sweeps: usize, rng: &mut Rng) -> Image<f32> {
    let mut labels = start.clone();
    for _ in 0..burn_in {
        gibbs_sweep(&mut labels, costs, lambda, 1.0, rng);
    }
    let mut count = Image::new(labels.width(), labels.height(), 0.0f32);
    for _ in 0..sweeps {
        gibbs_sweep(&mut labels, costs, lambda, 1.0, rng);
        for (x, y) in labels.coords() {
            count[(x, y)] += labels[(x, y)] as f32;
        }
    }
    count.map(|c| c / sweeps as f32)
}

/// C5. Mean field: every pixel holds `q_p ≈ p(L_p = 1)`. Start with each pixel alone,
/// `q_p = 1 / (1 + e^(D_p(1) − D_p(0)))`, then do `sweeps` sweeps (row by row, in place) of
/// `q_p = 1 / (1 + e^Δ)`, `Δ = D_p(1) − D_p(0) + λ · Σ_neighbours (1 − 2 q_q)`:
/// the local conditional with each neighbour replaced by its mean.
pub fn mean_field(costs: &Costs, lambda: f32, sweeps: usize) -> Image<f32> {
    let (w, h) = (costs.width(), costs.height());
    let mut q = costs.map(|[d0, d1]| 1.0 / (1.0 + (d1 - d0).exp()));
    for _ in 0..sweeps {
        for y in 0..h {
            for x in 0..w {
                let [d0, d1] = costs[(x, y)];
                // a neighbour with label 0 adds λ to E(1), one with label 1 adds λ to E(0)
                let mut delta = d1 - d0;
                for nb in neighbours(x, y, w, h) {
                    delta += lambda * (1.0 - 2.0 * q[nb]);
                }
                q[(x, y)] = 1.0 / (1.0 + delta.exp());
            }
        }
    }
    q
}
