// Model solution for the coding exercises of "Active contours (snakes)".

use imgcourse::{Border, Image};

/// A point `[x, y]` of the snake (a bead), in pixels.
pub type Point = [f32; 2];

/// C1. The internal force on bead `i` of a closed snake (slide "The math: the snake energy
/// and its force"): elastic `α (x[i-1] − 2 x[i] + x[i+1])` plus rigid
/// `−β (x[i-2] − 4 x[i-1] + 6 x[i] − 4 x[i+1] + x[i+2])`. Indices wrap around: after the
/// last bead comes the first.
pub fn internal_force(points: &[Point], i: usize, alpha: f32, beta: f32) -> Point {
    let n = points.len();
    let at = |k: isize| points[(i as isize + k).rem_euclid(n as isize) as usize];
    let mut force = [0.0; 2];
    for c in 0..2 {
        let (m2, m1, x, p1, p2) = (at(-2)[c], at(-1)[c], at(0)[c], at(1)[c], at(2)[c]);
        let elastic = alpha * (m1 - 2.0 * x + p1);
        let rigid = -beta * (m2 - 4.0 * m1 + 6.0 * x - 4.0 * p1 + p2);
        force[c] = elastic + rigid;
    }
    force
}

/// C2. The edge map `f = |∇(G_σ * I)|²` (slide "The image term"): blur, take the gradient
/// with central differences, square its length. Scaled so that its largest value is 1, so
/// that the image weight `w` does not depend on the contrast of the image.
pub fn edge_map(img: &Image<f32>, sigma: f32) -> Image<f32> {
    let smooth = gaussian_blur(img, sigma);
    let mut f = Image::new(img.width(), img.height(), 0.0);
    for (x, y) in img.coords() {
        let (x, y) = (x as isize, y as isize);
        let gx = (smooth.get_border(x + 1, y, Border::Clamp) - smooth.get_border(x - 1, y, Border::Clamp)) / 2.0;
        let gy = (smooth.get_border(x, y + 1, Border::Clamp) - smooth.get_border(x, y - 1, Border::Clamp)) / 2.0;
        f[(x as usize, y as usize)] = gx * gx + gy * gy;
    }
    let max = f.max();
    if max > 0.0 {
        f = f.map(|v| v / max);
    }
    f
}

/// C3. One explicit step of the snake (slide "Moving the snake: one step at a time"):
/// every bead moves by `τ × (internal force + w ∇f)`, with `∇f` read at the bead's
/// position by central differences of bilinear lookups. All forces use the old positions.
pub fn snake_step(points: &[Point], f: &Image<f32>, alpha: f32, beta: f32, w: f32, tau: f32) -> Vec<Point> {
    let mut new = Vec::with_capacity(points.len());
    for (i, p) in points.iter().enumerate() {
        let internal = internal_force(points, i, alpha, beta);
        let (x, y) = (p[0], p[1]);
        let fx = (bilinear(f, x + 1.0, y) - bilinear(f, x - 1.0, y)) / 2.0;
        let fy = (bilinear(f, x, y + 1.0) - bilinear(f, x, y - 1.0)) / 2.0;
        new.push([x + tau * (internal[0] + w * fx), y + tau * (internal[1] + w * fy)]);
    }
    new
}

// ---- Helpers from earlier units (filled in, not part of the exercise) ----

// `gaussian_blur(img, sigma)`, `bilinear(img, x, y)` and `circle(cx, cy, r, n)` (`n` beads on
// a circle) are in `src/common.rs`.
pub use crate::common::{bilinear, circle, gaussian_blur};
