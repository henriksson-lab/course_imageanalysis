// Coding exercises for "Active contours (snakes)". Replace each `todo!()` with your code,
// then check it with `cargo test -p snakes snakes::` and look at `cargo run -p snakes -- snakes`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};

/// A point `[x, y]` of the snake (a bead), in pixels.
pub type Point = [f32; 2];

/// C1. The internal force on bead `i` of a closed snake (slide "The math: the snake energy
/// and its force"): elastic `α (x[i-1] − 2 x[i] + x[i+1])` plus rigid
/// `−β (x[i-2] − 4 x[i-1] + 6 x[i] − 4 x[i+1] + x[i+2])`. Indices wrap around: after the
/// last bead comes the first.
///
/// Hint: `points[(i + n - 1) % n]` is the bead before bead `i`, also for `i = 0`.
/// Do the same sum for x (`[0]`) and for y (`[1]`).
pub fn internal_force(points: &[Point], i: usize, alpha: f32, beta: f32) -> Point {
    todo!("C1: internal_force")
}

/// C2. The edge map `f = |∇(G_σ * I)|²` (slide "The image term"): blur, take the gradient
/// with central differences, square its length. Scaled so that its largest value is 1, so
/// that the image weight `w` does not depend on the contrast of the image.
///
/// Hint: `gaussian_blur` (in `src/common.rs`) does the smoothing. The central difference in x is
/// `(I(x+1, y) − I(x−1, y)) / 2`; read the neighbours with `get_border(.., Border::Clamp)`.
/// Divide by `f.max()` at the end.
pub fn edge_map(img: &Image<f32>, sigma: f32) -> Image<f32> {
    todo!("C2: edge_map")
}

/// C3. One explicit step of the snake (slide "Moving the snake: one step at a time"):
/// every bead moves by `τ × (internal force + w ∇f)`, with `∇f` read at the bead's
/// position by central differences of bilinear lookups. All forces use the old positions.
///
/// Hint: `bilinear(f, x, y)` reads `f` between pixels. Build a new `Vec`, so that every
/// bead's force is computed from the old positions.
pub fn snake_step(points: &[Point], f: &Image<f32>, alpha: f32, beta: f32, w: f32, tau: f32) -> Vec<Point> {
    todo!("C3: snake_step")
}

// ---- Helpers from earlier units (filled in, not part of the exercise) ----

// `gaussian_blur(img, sigma)`, `bilinear(img, x, y)` and `circle(cx, cy, r, n)` (`n` beads on
// a circle) are in `src/common.rs`.
pub use crate::common::{bilinear, circle, gaussian_blur};
