// Coding exercises for "The random walker".
// Replace each `todo!()` with your code, then check it with `cargo test -p day29 random_walker::`
// and look at `cargo run -p day29 -- random-walker`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Image, Rng};

/// Seeds: 0 = free pixel, 1, 2, … = a seed of that label (1 = object, 2 = background).
pub type Seeds = Image<u32>;

/// The weights of the pixel graph (4-neighbours): `wx[(x, y)]` joins `(x, y)` and `(x + 1, y)`,
/// `wy[(x, y)]` joins `(x, y)` and `(x, y + 1)`. The last column of `wx` and the last row of
/// `wy` are 0 (no neighbour there).
#[derive(Clone, Debug)]
pub struct Weights {
    pub wx: Image<f32>,
    pub wy: Image<f32>,
}

impl Weights {
    /// The neighbours of `(x, y)` with the weights of the edges to them.
    pub fn neighbours(&self, x: usize, y: usize) -> Vec<((usize, usize), f32)> {
        let (w, h) = (self.wx.width(), self.wx.height());
        let mut n = Vec::with_capacity(4);
        if x + 1 < w {
            n.push(((x + 1, y), self.wx[(x, y)]));
        }
        if x > 0 {
            n.push(((x - 1, y), self.wx[(x - 1, y)]));
        }
        if y + 1 < h {
            n.push(((x, y + 1), self.wy[(x, y)]));
        }
        if y > 0 {
            n.push(((x, y - 1), self.wy[(x, y - 1)]));
        }
        n
    }
}

/// C1. The voltages of a chain of nodes joined by conductances `weights` (`weights[i]` joins
/// node `i` and `i + 1`). `seeds[i]` is `Some(voltage)` at a seed and `None` at a free node.
/// Between two seeds the voltage drops in proportion to the resistance `1 / w`; before the
/// first seed and after the last one it stays at that seed's value.
///
/// Hint: first collect the positions of the seeds. The resistance of edge `i` is `1.0 / weights[i]`.
pub fn chain_voltages(weights: &[f32], seeds: &[Option<f32>]) -> Vec<f32> {
    todo!("C1: chain_voltages")
}

/// C2. The contrast weights `w = exp(−(I_p − I_q)² / 2σ²)` of all 4-neighbour edges.
///
/// Hint: `Image::from_fn(w, h, |x, y| ...)`, with 0 where `x + 1 == w` (for `wx`) or
/// `y + 1 == h` (for `wy`).
pub fn edge_weights(img: &Image<f32>, sigma: f32) -> Weights {
    todo!("C2: edge_weights")
}

/// C3. One averaging sweep (Gauss–Seidel): every free pixel, row by row, takes the weighted
/// mean of its neighbours, `x_p = Σ w_pq x_q / Σ w_pq`, using the values already updated.
/// Seed pixels are not changed. Returns the largest change of any pixel.
///
/// Hint: `weights.neighbours(x, y)` (given) lists the neighbours with their weights.
pub fn averaging_sweep(x: &mut Image<f32>, seeds: &Seeds, weights: &Weights) -> f32 {
    todo!("C3: averaging_sweep")
}

/// C4a. `L_U v` for the free pixels (seed pixels of `v` are taken as 0, and the result is 0
/// there): `(L_U v)_p = d_p v_p − Σ_{q free} w_pq v_q`, with `d_p` the sum of all weights at `p`.
///
/// Hint: loop over `weights.neighbours(x, y)`: every neighbour adds `w · v_p`, and a free
/// neighbour also subtracts `w · v_q`.
pub fn apply_lu(v: &Image<f32>, seeds: &Seeds, weights: &Weights) -> Image<f32> {
    todo!("C4a: apply_lu")
}

/// The dot product of two images (all pixels), in `f64`.
pub fn dot(a: &Image<f32>, b: &Image<f32>) -> f64 {
    a.pixels().iter().zip(b.pixels()).map(|(&p, &q)| p as f64 * q as f64).sum()
}

/// C4b. The random walker: the probability that a walk from each pixel reaches a seed of
/// `label` before any other seed. Solves `L_U x_U = −B^T x_M` by conjugate gradients, from
/// x = 0, until the residual is below `tol` times its start (or `max_iterations`). Returns
/// the probabilities (1 and 0 at the seeds) and the number of iterations.
///
/// Hint: work with whole images for x, r and p, with 0 at the seeds; one `apply_lu` per
/// iteration. Then:
///   r = b, p = r, rs = r·r
///   repeat: Ap = L_U p; α = rs / (p·Ap); x += α p; r −= α Ap;
///           rs_new = r·r; p = r + (rs_new/rs) p; rs = rs_new
/// Compute the dot products in `f64`.
pub fn random_walker(seeds: &Seeds, weights: &Weights, label: u32, tol: f32, max_iterations: usize) -> (Image<f32>, usize) {
    todo!("C4b: random_walker")
}

/// C5. Estimate the same probability by simulation: release `walks` random walks from
/// `start`; each step goes to a neighbour with probability `w_pq / d_p`; a walk stops at
/// the first seed. Returns the fraction of walks that stop at a seed of `label`.
///
/// Hint: draw `u = rng.uniform() * total` and walk through the neighbours, subtracting
/// their weights until `u` is smaller than the next weight.
pub fn walk_estimate(seeds: &Seeds, weights: &Weights, label: u32, start: (usize, usize), walks: usize, rng: &mut Rng) -> f32 {
    todo!("C5: walk_estimate")
}
