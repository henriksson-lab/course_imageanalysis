// Model solution for the coding exercises of "Region-based segmentation: Chan–Vese".

use imgcourse::{Border, Image};

/// The knobs: the length weight `λ`, the area weight `ν`, the time step `Δt` and the width `ε`
/// of the soft step.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub lambda: f32,
    pub nu: f32,
    pub dt: f32,
    pub eps: f32,
}

/// C1. The two grey levels for a fixed curve (slide "Alternate: means, then move"): `c1` the
/// mean of the image where `φ < 0` (inside), `c2` the mean where `φ ≥ 0`. An empty region gets
/// the mean 0.
pub fn region_means(img: &Image<f32>, phi: &Image<f32>) -> (f32, f32) {
    let (mut sum_in, mut n_in, mut sum_out, mut n_out) = (0.0, 0, 0.0, 0);
    for (x, y) in img.coords() {
        if phi[(x, y)] < 0.0 {
            sum_in += img[(x, y)];
            n_in += 1;
        } else {
            sum_out += img[(x, y)];
            n_out += 1;
        }
    }
    let mean = |s: f32, n: usize| if n > 0 { s / n as f32 } else { 0.0 };
    (mean(sum_in, n_in), mean(sum_out, n_out))
}

/// C2. The Chan–Vese energy of a labelling (slide "The Chan–Vese energy", with `ν = 0`):
/// `Σ_inside (I − c1)² + Σ_outside (I − c2)² + λ · boundary`, with `c1`, `c2` the means of the
/// two regions and the boundary counted as the number of pairs of 4-neighbours with different
/// labels (in 1D: the number of boundary points).
pub fn energy(img: &Image<f32>, inside: &Image<bool>, lambda: f32) -> f32 {
    let phi = inside.map(|i| if i { -1.0 } else { 1.0 });
    let (c1, c2) = region_means(img, &phi);
    let mut e = 0.0;
    for (x, y) in img.coords() {
        let c = if inside[(x, y)] { c1 } else { c2 };
        e += (img[(x, y)] - c).powi(2);
        if x + 1 < img.width() && inside[(x, y)] != inside[(x + 1, y)] {
            e += lambda;
        }
        if y + 1 < img.height() && inside[(x, y)] != inside[(x, y + 1)] {
            e += lambda;
        }
    }
    e
}

/// C3. One gradient-descent step on φ for fixed `c1`, `c2` (slide "The math: gradient descent
/// on φ"), with `φ < 0` inside:
/// `φ + Δt δ_ε(φ) [λκ + ν + (I − c1)² − (I − c2)²]`, where `δ_ε(φ) = ε / (π (ε² + φ²))` is the
/// soft spike. `κ = ∇·(∇φ/|∇φ|)`, the divergence of the unit normal (the unit "Curves as level
/// sets"), here taken between neighbours so that a single pixel that sticks out is also
/// smoothed: the x part is `n(x+½) − n(x−½)` with `n(x+½) = D / √(D² + E² + 10⁻⁴)`, where
/// `D = φ(x+1, y) − φ(x, y)` and `E` is the slope in y there (the mean of the central y
/// differences at `x` and `x + 1`); the same for y.
pub fn chan_vese_step(img: &Image<f32>, phi: &Image<f32>, c1: f32, c2: f32, p: Params) -> Image<f32> {
    let Params { lambda, nu, dt, eps } = p;
    phi.map_xy(|x, y, v| {
        let (xi, yi) = (x as isize, y as isize);
        let at = |dx: isize, dy: isize| phi.get_border(xi + dx, yi + dy, Border::Clamp);
        // the unit normal component across the edge between two neighbours
        let n = |d: f32, e: f32| d / (d * d + e * e + 1e-4).sqrt();
        let ey = |dx: isize| (at(dx, 1) - at(dx, -1)) / 2.0; // slope in y at column x + dx
        let ex = |dy: isize| (at(1, dy) - at(-1, dy)) / 2.0; // slope in x at row y + dy
        let kx = n(at(1, 0) - v, (ey(0) + ey(1)) / 2.0) - n(v - at(-1, 0), (ey(0) + ey(-1)) / 2.0);
        let ky = n(at(0, 1) - v, (ex(0) + ex(1)) / 2.0) - n(v - at(0, -1), (ex(0) + ex(-1)) / 2.0);
        let kappa = kx + ky;
        let i = img[(x, y)];
        let delta = eps / (std::f32::consts::PI * (eps * eps + v * v));
        v + dt * delta * (lambda * kappa + nu + (i - c1).powi(2) - (i - c2).powi(2))
    })
}

/// C4. Chan–Vese (slide "Alternate: means, then move"): `iterations` times, compute the means
/// (C1), then take one step (C3). Returns the final φ and the means used in the last step.
pub fn chan_vese(img: &Image<f32>, phi0: &Image<f32>, p: Params, iterations: usize) -> (Image<f32>, f32, f32) {
    let mut phi = phi0.clone();
    let (mut c1, mut c2) = (0.0, 0.0);
    for _ in 0..iterations {
        (c1, c2) = region_means(img, &phi);
        phi = chan_vese_step(img, &phi, c1, c2, p);
    }
    (phi, c1, c2)
}
