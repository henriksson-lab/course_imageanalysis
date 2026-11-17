// Coding exercises for "Bayesian image denoising with an Ising prior".
// Replace each `todo!()` with your code, then check it with `cargo test -p ising mrf_denoise::`
// and look at `cargo run -p ising -- mrf-denoise`.
#![allow(unused_variables, unused_imports)]

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
///
/// Hint: `costs[(x, y)][labels[(x, y)] as usize]` is the data cost of the current label. Add
/// λ for the right neighbour and for the one below when their label differs.
pub fn energy(labels: &Labels, costs: &Costs, lambda: f32) -> f32 {
    todo!("C1: energy")
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
///
/// Hint: `local_energy` (given) is the energy of one pixel with a label, given its
/// neighbours. Use a `changed` flag per sweep.
pub fn icm(start: &Labels, costs: &Costs, lambda: f32, order: &[(usize, usize)]) -> (Labels, usize) {
    todo!("C2: icm")
}

/// C3. One Gibbs sweep at temperature `t`, row by row: draw each label from its local
/// conditional, label 1 with probability `1 / (1 + e^(Δ/T))`, `Δ = E(L_p = 1) − E(L_p = 0)`
/// with the neighbours as they are.
///
/// Hint: `Δ = local_energy(.., 1) − local_energy(.., 0)`; `rng.uniform()` is in [0, 1).
pub fn gibbs_sweep(labels: &mut Labels, costs: &Costs, lambda: f32, t: f32, rng: &mut Rng) {
    todo!("C3: gibbs_sweep")
}

/// C4. The marginals `p(L_p = 1 | y)` by sampling at `T = 1`: start from `start`, do
/// `burn_in` Gibbs sweeps, then `sweeps` more, and return for every pixel the fraction of
/// those `sweeps` in which it was 1.
///
/// Hint: run `gibbs_sweep` at `t = 1.0`; add each label (0 or 1) to a counting image.
pub fn marginals(start: &Labels, costs: &Costs, lambda: f32, burn_in: usize, sweeps: usize, rng: &mut Rng) -> Image<f32> {
    todo!("C4: marginals")
}

/// C5. Mean field: every pixel holds `q_p ≈ p(L_p = 1)`. Start with each pixel alone,
/// `q_p = 1 / (1 + e^(D_p(1) − D_p(0)))`, then do `sweeps` sweeps (row by row, in place) of
/// `q_p = 1 / (1 + e^Δ)`, `Δ = D_p(1) − D_p(0) + λ · Σ_neighbours (1 − 2 q_q)`:
/// the local conditional with each neighbour replaced by its mean.
///
/// Hint: `neighbours(x, y, w, h)` (given) lists the neighbours inside the image.
pub fn mean_field(costs: &Costs, lambda: f32, sweeps: usize) -> Image<f32> {
    todo!("C5: mean_field")
}
