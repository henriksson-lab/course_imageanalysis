// Coding exercises for "Sampling, temperature and the phase transition".
// Replace each `todo!()` with your code, then check it with `cargo test -p mcmc`
// and look at `cargo run -p mcmc`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image, Rng};

/// A lattice of spins, `+1` or `-1`, one per pixel. All lattices here are periodic (a torus):
/// the right neighbour of the last column is the first column, and likewise for rows.
pub type Spins = Image<i8>;

/// Random spins: each `+1` or `-1` with probability one half.
pub fn random_spins(width: usize, height: usize, rng: &mut Rng) -> Spins {
    Image::from_fn(width, height, |_, _| if rng.uniform() < 0.5 { 1 } else { -1 })
}

/// C1. The local field `u = J · (sum of the 4 neighbouring spins) + h` at `(x, y)`, with
/// periodic neighbours. This is the neighbours' vote for `+1`.
///
/// Hint: `s.get_border(x as isize + 1, y as isize, Border::Wrap)` reads a neighbour with
/// wrap-around; spins are `i8`, so use `as f32`.
pub fn local_field(s: &Spins, x: usize, y: usize, j: f32, h: f32) -> f32 {
    todo!("C1: local_field")
}

/// C2. One sweep of the Gibbs sampler: visit every pixel row by row and draw it anew,
/// `+1` with probability `1 / (1 + e^(−2u/T))` (slide "Two samplers"), else `−1`.
///
/// Hint: change `s` in place: later pixels of the same sweep see the new values.
/// `rng.uniform()` is a random number in [0, 1).
pub fn gibbs_sweep(s: &mut Spins, j: f32, h: f32, t: f32, rng: &mut Rng) {
    todo!("C2: gibbs_sweep")
}

/// C3. One sweep of Metropolis: visit every pixel row by row and propose to flip it. The
/// energy change of the flip is `ΔE = 2 · s · u`; accept if `ΔE ≤ 0`, otherwise with
/// probability `e^(−ΔE/T)`. Returns the number of accepted flips.
///
/// Hint: flipping is `s[(x, y)] = -s[(x, y)]`.
pub fn metropolis_sweep(s: &mut Spins, j: f32, h: f32, t: f32, rng: &mut Rng) -> usize {
    todo!("C3: metropolis_sweep")
}

/// C4. The energy per spin, `E/n` with `E = −J Σ s_p s_q − h Σ s_p` (periodic, every pair of
/// 4-neighbours once), and the magnetisation `m = (1/n) Σ s_p`.
pub fn energy_and_magnetisation(s: &Spins, j: f32, h: f32) -> (f32, f32) {
    todo!("C4: energy_and_magnetisation")
}

/// C5. Simulated annealing with a geometric schedule: `sweeps` Metropolis sweeps (`h = 0`) at
/// the temperatures `T_k = t0 · r^k`, `k = 0 .. sweeps − 1`, with `r` chosen so that the last
/// one is `t1`. Returns the temperatures used.
///
/// Hint: `r = (t1 / t0).powf(1.0 / (sweeps - 1) as f32)`; multiply `t` by `r` after each sweep.
pub fn anneal(s: &mut Spins, j: f32, t0: f32, t1: f32, sweeps: usize, rng: &mut Rng) -> Vec<f32> {
    todo!("C5: anneal")
}
