// Model solution for the coding exercises of "The random walker".

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
pub fn chain_voltages(weights: &[f32], seeds: &[Option<f32>]) -> Vec<f32> {
    let n = seeds.len();
    let seeded: Vec<usize> = (0..n).filter(|&i| seeds[i].is_some()).collect();
    let mut v = vec![0.0; n];
    let (first, last) = (seeded[0], seeded[seeded.len() - 1]);
    for i in 0..=first {
        v[i] = seeds[first].unwrap();
    }
    for i in last..n {
        v[i] = seeds[last].unwrap();
    }
    for pair in seeded.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let (va, vb) = (seeds[a].unwrap(), seeds[b].unwrap());
        let total: f32 = (a..b).map(|i| 1.0 / weights[i]).sum();
        let mut r = 0.0; // resistance from a to node i
        for i in a..b {
            v[i] = va + (vb - va) * r / total;
            r += 1.0 / weights[i];
        }
        v[b] = vb;
    }
    v
}

/// C2. The contrast weights `w = exp(−(I_p − I_q)² / 2σ²)` of all 4-neighbour edges.
pub fn edge_weights(img: &Image<f32>, sigma: f32) -> Weights {
    let (w, h) = (img.width(), img.height());
    let weight = |a: f32, b: f32| (-(a - b).powi(2) / (2.0 * sigma * sigma)).exp();
    let wx = Image::from_fn(w, h, |x, y| if x + 1 < w { weight(img[(x, y)], img[(x + 1, y)]) } else { 0.0 });
    let wy = Image::from_fn(w, h, |x, y| if y + 1 < h { weight(img[(x, y)], img[(x, y + 1)]) } else { 0.0 });
    Weights { wx, wy }
}

/// C3. One averaging sweep (Gauss–Seidel): every free pixel, row by row, takes the weighted
/// mean of its neighbours, `x_p = Σ w_pq x_q / Σ w_pq`, using the values already updated.
/// Seed pixels are not changed. Returns the largest change of any pixel.
pub fn averaging_sweep(x: &mut Image<f32>, seeds: &Seeds, weights: &Weights) -> f32 {
    let mut largest = 0.0f32;
    for y in 0..x.height() {
        for px in 0..x.width() {
            if seeds[(px, y)] != 0 {
                continue;
            }
            let (mut sum, mut total) = (0.0, 0.0);
            for (q, w) in weights.neighbours(px, y) {
                sum += w * x[q];
                total += w;
            }
            if total > 0.0 {
                let new = sum / total;
                largest = largest.max((new - x[(px, y)]).abs());
                x[(px, y)] = new;
            }
        }
    }
    largest
}

/// C4a. `L_U v` for the free pixels (seed pixels of `v` are taken as 0, and the result is 0
/// there): `(L_U v)_p = d_p v_p − Σ_{q free} w_pq v_q`, with `d_p` the sum of all weights at `p`.
pub fn apply_lu(v: &Image<f32>, seeds: &Seeds, weights: &Weights) -> Image<f32> {
    Image::from_fn(v.width(), v.height(), |x, y| {
        if seeds[(x, y)] != 0 {
            return 0.0;
        }
        let mut out = 0.0;
        for (q, w) in weights.neighbours(x, y) {
            out += w * v[(x, y)];
            if seeds[q] == 0 {
                out -= w * v[q];
            }
        }
        out
    })
}

/// The dot product of two images (all pixels), in `f64`.
pub fn dot(a: &Image<f32>, b: &Image<f32>) -> f64 {
    a.pixels().iter().zip(b.pixels()).map(|(&p, &q)| p as f64 * q as f64).sum()
}

/// C4b. The random walker: the probability that a walk from each pixel reaches a seed of
/// `label` before any other seed. Solves `L_U x_U = −B^T x_M` by conjugate gradients, from
/// x = 0, until the residual is below `tol` times its start (or `max_iterations`). Returns
/// the probabilities (1 and 0 at the seeds) and the number of iterations.
pub fn random_walker(seeds: &Seeds, weights: &Weights, label: u32, tol: f32, max_iterations: usize) -> (Image<f32>, usize) {
    let fixed = seeds.map(|s| if s == label { 1.0 } else { 0.0 }); // the values at the seeds
    // right side b_p = Σ_{q seed} w_pq x_q for free p: what the seeds pull with
    let b = Image::from_fn(seeds.width(), seeds.height(), |x, y| {
        if seeds[(x, y)] != 0 {
            return 0.0;
        }
        weights.neighbours(x, y).iter().filter(|(q, _)| seeds[*q] != 0).map(|&(q, w)| w * fixed[q]).sum()
    });
    // conjugate gradients for L_U x = b, starting from x = 0 (so r = b)
    let mut x = Image::new(seeds.width(), seeds.height(), 0.0f32);
    let mut r = b.clone();
    let mut p = r.clone();
    let mut rs = dot(&r, &r);
    let stop = (tol as f64).powi(2) * rs;
    let mut iterations = 0;
    while iterations < max_iterations && rs > stop && rs > 0.0 {
        let ap = apply_lu(&p, seeds, weights);
        let alpha = (rs / dot(&p, &ap)) as f32;
        x = x.zip_map(&p, |a, b| a + alpha * b);
        r = r.zip_map(&ap, |a, b| a - alpha * b);
        let rs_new = dot(&r, &r);
        let beta = (rs_new / rs) as f32;
        p = r.zip_map(&p, |a, b| a + beta * b);
        rs = rs_new;
        iterations += 1;
    }
    // put the seed values back
    (x.map_xy(|px, py, v| if seeds[(px, py)] != 0 { fixed[(px, py)] } else { v }), iterations)
}

/// C5. Estimate the same probability by simulation: release `walks` random walks from
/// `start`; each step goes to a neighbour with probability `w_pq / d_p`; a walk stops at
/// the first seed. Returns the fraction of walks that stop at a seed of `label`.
pub fn walk_estimate(seeds: &Seeds, weights: &Weights, label: u32, start: (usize, usize), walks: usize, rng: &mut Rng) -> f32 {
    let mut hits = 0;
    for _ in 0..walks {
        let mut p = start;
        while seeds[p] == 0 {
            let n = weights.neighbours(p.0, p.1);
            let total: f32 = n.iter().map(|(_, w)| w).sum();
            // choose neighbour k with probability w_k / total
            let mut u = rng.uniform() * total;
            let mut next = n[n.len() - 1].0;
            for &(q, w) in &n {
                if u < w {
                    next = q;
                    break;
                }
                u -= w;
            }
            p = next;
        }
        if seeds[p] == label {
            hits += 1;
        }
    }
    hits as f32 / walks as f32
}
