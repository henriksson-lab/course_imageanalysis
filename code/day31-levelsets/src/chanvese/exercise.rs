// Coding exercises for "Region-based segmentation: Chan–Vese".
// Replace each `todo!()` with your code, then check it with `cargo test -p day31 chanvese::` and look at
// `cargo run -p day31 -- chanvese`.
#![allow(unused_variables, unused_imports)]

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
///
/// Hint: one pass over `img.coords()`, with a sum and a count for each region.
pub fn region_means(img: &Image<f32>, phi: &Image<f32>) -> (f32, f32) {
    todo!("C1: region_means")
}

/// C2. The Chan–Vese energy of a labelling (slide "The Chan–Vese energy", with `ν = 0`):
/// `Σ_inside (I − c1)² + Σ_outside (I − c2)² + λ · boundary`, with `c1`, `c2` the means of the
/// two regions and the boundary counted as the number of pairs of 4-neighbours with different
/// labels (in 1D: the number of boundary points).
///
/// Hint: `region_means(img, &inside.map(|i| if i { -1.0 } else { 1.0 }))` gives the means. Count
/// a pair only once: look at the right and the lower neighbour of every pixel.
pub fn energy(img: &Image<f32>, inside: &Image<bool>, lambda: f32) -> f32 {
    todo!("C2: energy")
}

/// C3. One gradient-descent step on φ for fixed `c1`, `c2` (slide "The math: gradient descent
/// on φ"), with `φ < 0` inside:
/// `φ + Δt δ_ε(φ) [λκ + ν + (I − c1)² − (I − c2)²]`, where `δ_ε(φ) = ε / (π (ε² + φ²))` is the
/// soft spike. `κ = ∇·(∇φ/|∇φ|)`, the divergence of the unit normal (the unit "Curves as level
/// sets"), here taken between neighbours so that a single pixel that sticks out is also
/// smoothed: the x part is `n(x+½) − n(x−½)` with `n(x+½) = D / √(D² + E² + 10⁻⁴)`, where
/// `D = φ(x+1, y) − φ(x, y)` and `E` is the slope in y there (the mean of the central y
/// differences at `x` and `x + 1`); the same for y.
///
/// Hint: write the closures `n(d, e)` and `at(dx, dy)` first; then `κ` is two differences of `n`.
pub fn chan_vese_step(img: &Image<f32>, phi: &Image<f32>, c1: f32, c2: f32, p: Params) -> Image<f32> {
    todo!("C3: chan_vese_step")
}

/// C4. Chan–Vese (slide "Alternate: means, then move"): `iterations` times, compute the means
/// (C1), then take one step (C3). Returns the final φ and the means used in the last step.
///
/// Hint: `(c1, c2) = region_means(img, &phi);` works as an assignment to two variables.
pub fn chan_vese(img: &Image<f32>, phi0: &Image<f32>, p: Params, iterations: usize) -> (Image<f32>, f32, f32) {
    todo!("C4: chan_vese")
}
