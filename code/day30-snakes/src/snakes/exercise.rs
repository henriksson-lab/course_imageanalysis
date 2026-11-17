// Coding exercises for "Active contours (snakes)". Replace each `todo!()` with your code,
// then check it with `cargo test -p day30 snakes::` and look at `cargo run -p day30 -- snakes`.
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
/// Hint: `gaussian_blur` (below) does the smoothing. The central difference in x is
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

/// Gaussian blur of width `sigma` in two passes (unit "Filtering & convolution"), copying
/// the edge outside the image. `sigma = 0` returns a copy.
pub fn gaussian_blur(img: &Image<f32>, sigma: f32) -> Image<f32> {
    if sigma <= 0.0 {
        return img.clone();
    }
    let r = (3.0 * sigma).ceil() as isize;
    let mut w: Vec<f32> = (-r..=r).map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let total: f32 = w.iter().sum();
    w.iter_mut().for_each(|v| *v /= total);
    let pass = |src: &Image<f32>, along_x: bool| {
        src.map_xy(|x, y, _| {
            let mut sum = 0.0;
            for d in -r..=r {
                let (xx, yy) = if along_x { (x as isize + d, y as isize) } else { (x as isize, y as isize + d) };
                sum += w[(d + r) as usize] * src.get_border(xx, yy, Border::Clamp);
            }
            sum
        })
    };
    pass(&pass(img, true), false)
}

/// The image value at a position between pixels, by bilinear interpolation (unit "Images as
/// arrays"); outside the image the nearest edge pixel is used.
pub fn bilinear(img: &Image<f32>, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (x0, y0) = (x0 as isize, y0 as isize);
    let p = |dx: isize, dy: isize| img.get_border(x0 + dx, y0 + dy, Border::Clamp);
    (1.0 - fx) * (1.0 - fy) * p(0, 0) + fx * (1.0 - fy) * p(1, 0) + (1.0 - fx) * fy * p(0, 1) + fx * fy * p(1, 1)
}

/// `n` beads on a circle of radius `r` around `(cx, cy)`.
pub fn circle(cx: f32, cy: f32, r: f32, n: usize) -> Vec<Point> {
    (0..n)
        .map(|i| {
            let t = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
            [cx + r * t.cos(), cy + r * t.sin()]
        })
        .collect()
}
