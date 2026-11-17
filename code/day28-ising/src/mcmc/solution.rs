// Model solution for the coding exercises of "Sampling, temperature and the phase transition".

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
pub fn local_field(s: &Spins, x: usize, y: usize, j: f32, h: f32) -> f32 {
    let (x, y) = (x as isize, y as isize);
    let mut sum = 0.0;
    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        sum += s.get_border(x + dx, y + dy, Border::Wrap) as f32;
    }
    j * sum + h
}

/// C2. One sweep of the Gibbs sampler: visit every pixel row by row and draw it anew,
/// `+1` with probability `1 / (1 + e^(−2u/T))` (slide "Two samplers"), else `−1`.
pub fn gibbs_sweep(s: &mut Spins, j: f32, h: f32, t: f32, rng: &mut Rng) {
    for y in 0..s.height() {
        for x in 0..s.width() {
            let u = local_field(s, x, y, j, h);
            let p_plus = 1.0 / (1.0 + (-2.0 * u / t).exp());
            s[(x, y)] = if rng.uniform() < p_plus { 1 } else { -1 };
        }
    }
}

/// C3. One sweep of Metropolis: visit every pixel row by row and propose to flip it. The
/// energy change of the flip is `ΔE = 2 · s · u`; accept if `ΔE ≤ 0`, otherwise with
/// probability `e^(−ΔE/T)`. Returns the number of accepted flips.
pub fn metropolis_sweep(s: &mut Spins, j: f32, h: f32, t: f32, rng: &mut Rng) -> usize {
    let mut accepted = 0;
    for y in 0..s.height() {
        for x in 0..s.width() {
            let delta_e = 2.0 * s[(x, y)] as f32 * local_field(s, x, y, j, h);
            if delta_e <= 0.0 || rng.uniform() < (-delta_e / t).exp() {
                s[(x, y)] = -s[(x, y)];
                accepted += 1;
            }
        }
    }
    accepted
}

/// C4. The energy per spin, `E/n` with `E = −J Σ s_p s_q − h Σ s_p` (periodic, every pair of
/// 4-neighbours once), and the magnetisation `m = (1/n) Σ s_p`.
pub fn energy_and_magnetisation(s: &Spins, j: f32, h: f32) -> (f32, f32) {
    let mut pairs = 0.0;
    let mut sum = 0.0;
    for (x, y) in s.coords() {
        let sp = s[(x, y)] as f32;
        sum += sp;
        // the right and the lower neighbour: every pair once
        pairs += sp * s.get_border(x as isize + 1, y as isize, Border::Wrap) as f32;
        pairs += sp * s.get_border(x as isize, y as isize + 1, Border::Wrap) as f32;
    }
    let n = (s.width() * s.height()) as f32;
    ((-j * pairs - h * sum) / n, sum / n)
}

/// C5. Simulated annealing with a geometric schedule: `sweeps` Metropolis sweeps (`h = 0`) at
/// the temperatures `T_k = t0 · r^k`, `k = 0 .. sweeps − 1`, with `r` chosen so that the last
/// one is `t1`. Returns the temperatures used.
pub fn anneal(s: &mut Spins, j: f32, t0: f32, t1: f32, sweeps: usize, rng: &mut Rng) -> Vec<f32> {
    // slide "The math: cooling schedules": r = (T1/T0)^(1/(K−1)) reaches t1 at the last sweep
    let r = if sweeps > 1 { (t1 / t0).powf(1.0 / (sweeps - 1) as f32) } else { 1.0 };
    let mut temperatures = Vec::with_capacity(sweeps);
    let mut t = t0;
    for _ in 0..sweeps {
        metropolis_sweep(s, j, 0.0, t, rng);
        temperatures.push(t);
        t *= r;
    }
    temperatures
}
