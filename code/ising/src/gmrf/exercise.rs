// Coding exercises for "Continuous MRFs: quadratic smoothing and total variation".
// Replace each `todo!()` with your code, then check it with `cargo test -p ising gmrf::`
// and look at `cargo run -p ising -- gmrf`.
#![allow(unused_variables, unused_imports)]
// variation".

use imgcourse::Image;

/// The penalty `ρ(d)` on a difference `d` between neighbours.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Penalty {
    /// `d²/2`
    Quadratic,
    /// `|d|` (total variation)
    Tv,
    /// `d²/2` for `|d| ≤ δ`, `δ(|d| − δ/2)` beyond
    Huber(f32),
}

/// C1a. The penalty `ρ(d)` on a difference `d` between neighbours.
///
/// Hint: `match rho { Penalty::Quadratic => ..., Penalty::Tv => ..., Penalty::Huber(delta) => ... }`.
pub fn penalty(d: f32, rho: Penalty) -> f32 {
    todo!("C1a: penalty")
}

/// C1b. The energy of a 1D signal `x` for the data `y`:
/// `E = ½ Σ_p (x_p − y_p)² + λ Σ_p ρ(x_{p+1} − x_p)`.
pub fn energy_1d(x: &[f32], y: &[f32], lambda: f32, rho: Penalty) -> f32 {
    todo!("C1b: energy_1d")
}

/// C2. Quadratic smoothing of a 1D signal: solve `(I + λL) x = y`, with the path Laplacian
/// `L` (diagonal 1, 2, …, 2, 1, and −1 next to it), by elimination for a tridiagonal system.
///
/// Hint: forward pass: for p = 1, 2, …, remove x_{p−1} from row p (keep the new diagonal and
/// right side); backward pass: x_{n−1} first, then x_p from x_{p+1}. Check on (0, 3, 0) by hand.
pub fn quadratic_1d(y: &[f32], lambda: f32) -> Vec<f32> {
    todo!("C2: quadratic_1d")
}

/// C3. Total-variation smoothing of a 1D signal: minimise `½ Σ (x_p − y_p)² + λ Σ |x_{p+1} − x_p|`.
/// Each pair `(p, p+1)` gets a number `z_p` between `−λ` and `λ` (`λ · g_p` of the math slide);
/// then `x_p = y_p − (z_{p−1} − z_p)` (with `z = 0` beyond the ends). Repeat: compute `x`,
/// then move every `z_p` by `¼ (x_{p+1} − x_p)` and clip it to `[−λ, λ]`.
///
/// Hint: `z` has `n − 1` entries; `.clamp(-lambda, lambda)` clips a number.
pub fn tv_1d(y: &[f32], lambda: f32, iterations: usize) -> Vec<f32> {
    todo!("C3: tv_1d")
}

/// C4. Quadratic smoothing of an image (4-neighbours): solve `(I + λL) x = y` by Gauss–Seidel
/// sweeps. Each sweep sets, pixel by pixel in place, `x_p = (y_p + λ Σ_q x_q) / (1 + λ · deg_p)`
/// (the sum over the neighbours inside the image, `deg_p` their number), starting from `y`.
///
/// Hint: `x.contains(qx, qy)` takes signed coordinates and tells whether a neighbour exists.
pub fn quadratic_2d(y: &Image<f32>, lambda: f32, sweeps: usize) -> Image<f32> {
    todo!("C4: quadratic_2d")
}

/// C5. Total-variation smoothing of an image, `½ Σ (x_p − y_p)² + λ Σ |x_p − x_q|` over the
/// pairs of 4-neighbours: C3 with one `z` per horizontal pair (`zx`, between `(x, y)` and
/// `(x+1, y)`) and one per vertical pair (`zy`), and step `1/8` instead of `1/4`.
///
/// Hint: write a small function or closure that computes x from `zx` and `zy`, as in C3.
pub fn tv_2d(y: &Image<f32>, lambda: f32, iterations: usize) -> Image<f32> {
    todo!("C5: tv_2d")
}
