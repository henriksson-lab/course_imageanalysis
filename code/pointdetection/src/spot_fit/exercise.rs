// Coding exercises for "Sub-pixel localisation".
// Replace each `todo!()` with your code, then check it with `cargo test -p pointdetection spot_fit::`
// and look at `cargo run -p pointdetection -- spot-fit`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// C1. The centroid of a patch after subtracting `background` from every pixel:
/// `x̂ = Σ x (I − b) / Σ (I − b)`, and the same for `y`. Returns `[x̂, ŷ]` in the patch's
/// pixel coordinates.
///
/// Hint: three running sums over `patch.coords()`: of `x * v`, of `y * v` and of `v`,
/// with `v = patch[(x, y)] - background`.
pub fn centroid(patch: &Image<f32>, background: f32) -> [f32; 2] {
    todo!("C1: centroid")
}

/// C2a. The vertex of the parabola through `(−1, cm)`, `(0, c0)`, `(1, cp)`: the offset of the
/// peak from the middle sample.
///
/// Hint: the formula of the Background table; return 0 if the denominator is 0.
pub fn parabola_vertex(cm: f32, c0: f32, cp: f32) -> f32 {
    todo!("C2a: parabola_vertex")
}

/// C2b. The same through the logarithms of the three values: exact for a Gaussian.
/// Values that are not positive are raised to a tiny positive number first.
///
/// Hint: call `parabola_vertex` with `c.max(1e-6).ln()` of each value.
pub fn log_parabola_vertex(cm: f32, c0: f32, cp: f32) -> f32 {
    todo!("C2b: log_parabola_vertex")
}

/// C2c. The three-point estimate in 2D: find the brightest pixel, subtract `background` from it
/// and its four neighbours, and apply the (log-)parabola along x and along y. At the patch edge
/// the brightest pixel itself is returned.
///
/// Hint: the brightest pixel is
/// `patch.coords().max_by(|&a, &b| patch[a].total_cmp(&patch[b])).unwrap()`.
pub fn three_point(patch: &Image<f32>, background: f32, log: bool) -> [f32; 2] {
    todo!("C2c: three_point")
}

/// The spot shape of peak 1 at pixel `(x, y)`: `exp(−((x − x0)² + (y − y0)²) / 2s²)`.
pub fn shape(x: usize, y: usize, x0: f32, y0: f32, s: f32) -> f32 {
    let d2 = (x as f32 - x0).powi(2) + (y as f32 - y0).powi(2);
    (-d2 / (2.0 * s * s)).exp()
}

/// C3. With the position `(x0, y0)` and width `s` known, the amplitude `A` and background `B` of
/// the model `B + A · shape` that fit the patch best (least squares): solve the 2 × 2 normal
/// equations `[Σg² Σg; Σg n] [A; B] = [ΣgI; ΣI]`. Returns `(A, B)`.
///
/// Hint: four sums (Σg², Σg, ΣgI, ΣI) with `g = shape(x, y, x0, y0, s)`;
/// solve the 2 × 2 system by Cramer's rule: with `det = Σg² n − (Σg)²`,
/// `A = (n ΣgI − Σg ΣI) / det` and `B = (Σg² ΣI − Σg ΣgI) / det`.
pub fn fit_amplitude_background(patch: &Image<f32>, x0: f32, y0: f32, s: f32) -> (f32, f32) {
    todo!("C3: fit_amplitude_background")
}

/// C4. One Gauss–Newton step for the position, with `A`, `B` and `s` fixed, for each axis
/// separately: `Δx0 = Σ Jx r / Σ Jx²` with the residual `r = I − (B + A g)` and the derivative
/// `Jx = ∂(model)/∂x0 = A g (x − x0) / s²` (and the same for y). An axis with `Σ J² = 0` (a
/// patch one pixel high) does not move. Returns `[Δx0, Δy0]`.
///
/// Hint: four sums: Σ Jx r, Σ Jx², Σ Jy r, Σ Jy².
pub fn position_step(patch: &Image<f32>, x0: f32, y0: f32, s: f32, a: f32, b: f32) -> [f32; 2] {
    todo!("C4: position_step")
}

/// The result of a Gaussian fit.
#[derive(Clone, Copy, Debug)]
pub struct Fit {
    pub x: f32,
    pub y: f32,
    pub amplitude: f32,
    pub background: f32,
}

/// C5. Fit position, amplitude and background of one spot of known width `s`, starting at
/// `start`: `iterations` times, first C3 (A and B at the current position), then C4 (move the
/// position, by at most 1 pixel per step, so that a bad start cannot throw it far away).
///
/// Hint: `let [mut x0, mut y0] = start;` and a loop; `dx.clamp(-1.0, 1.0)` limits a step.
pub fn fit_spot(patch: &Image<f32>, start: [f32; 2], s: f32, iterations: usize) -> Fit {
    todo!("C5: fit_spot")
}
