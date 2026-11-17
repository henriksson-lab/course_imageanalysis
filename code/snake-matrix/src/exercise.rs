// Coding exercises for "The snake as a linear system". Replace each `todo!()` with your code,
// then check it with `cargo test -p snake-matrix` and look at `cargo run -p snake-matrix`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{linalg, Border, Image};

/// C1. The matrix `A` of the internal forces (slide "All beads at once: a matrix"): force
/// `= −A x`. Row `i` has `2α + 6β` on the diagonal, `−α − 4β` at `i ± 1` and `β` at `i ± 2`,
/// with the columns taken modulo `n` (the curve is closed). Entries that land on the same
/// column (possible for `n < 5`) add up.
///
/// Hint: start from `vec![vec![0.0; n]; n]`; for each row `i` and each offset `d` in −2..=2,
/// add the weight at column `(i as isize + d).rem_euclid(n as isize) as usize`.
pub fn snake_matrix(n: usize, alpha: f64, beta: f64) -> Vec<Vec<f64>> {
    todo!("C1: snake_matrix")
}

/// C2. One explicit step for one coordinate (slide "Why the explicit step explodes"):
/// `x_new = x − τ (A x − f)`, with `f` the image force on every bead.
///
/// Hint: first `(A x)[i]`, the sum over `j` of `a[i][j] * x[j]`, then the step.
pub fn explicit_step(a: &[Vec<f64>], x: &[f64], f: &[f64], tau: f64) -> Vec<f64> {
    todo!("C2: explicit_step")
}

/// C3. "Factor once" (slide "The semi-implicit step"): the inverse of `A + γI`, `γ = 1/τ`,
/// computed column by column: column `j` solves `(A + γI) c = e_j` (`e_j` = 1 at `j`, else 0).
///
/// Hint: copy `a`, add `γ` on the diagonal, then call `linalg::solve(&m, &e)` once per column.
pub fn semi_implicit_inverse(a: &[Vec<f64>], tau: f64) -> Vec<Vec<f64>> {
    todo!("C3: semi_implicit_inverse")
}

/// C4. One semi-implicit step for one coordinate: `x_new = (A + γI)⁻¹ (γ x + f)`, with the
/// inverse from C3.
pub fn semi_implicit_step(inv: &[Vec<f64>], x: &[f64], f: &[f64], tau: f64) -> Vec<f64> {
    todo!("C4: semi_implicit_step")
}

/// C5. Gradient vector flow (slide "Gradient vector flow"): start from `v = g` (the edge
/// force, `(gx, gy)`) and repeat `iterations` times, for both components,
/// `v ← v + Δt [ μ ∇²v − |g|² (v − g) ]`, with the 5-point Laplacian (neighbours outside the
/// image: copy the edge). Every step reads only the old field.
///
/// Hint: `map_xy` builds a new image from the old one. Compute `|g|²` once before the loop.
pub fn gvf(gx: &Image<f32>, gy: &Image<f32>, mu: f32, dt: f32, iterations: usize) -> (Image<f32>, Image<f32>) {
    todo!("C5: gvf")
}

// ---- Helpers from earlier units (filled in, not part of the exercise) ----

/// Gaussian blur of width `sigma` in two passes, copying the edge outside the image.
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

/// The gradient `(∂/∂x, ∂/∂y)` of an image by central differences, copying the edge.
pub fn gradient(img: &Image<f32>) -> (Image<f32>, Image<f32>) {
    let d = |dx: isize, dy: isize| {
        img.map_xy(|x, y, _| {
            let (x, y) = (x as isize, y as isize);
            (img.get_border(x + dx, y + dy, Border::Clamp) - img.get_border(x - dx, y - dy, Border::Clamp)) / 2.0
        })
    };
    (d(1, 0), d(0, 1))
}

/// The edge map of the snakes unit, `|∇(G_σ * I)|²`, scaled to largest value 1.
pub fn edge_map(img: &Image<f32>, sigma: f32) -> Image<f32> {
    let (gx, gy) = gradient(&gaussian_blur(img, sigma));
    let f = gx.zip_map(&gy, |a, b| a * a + b * b);
    let max = f.max();
    f.map(|v| if max > 0.0 { v / max } else { v })
}

/// The image value at a position between pixels, by bilinear interpolation; outside the
/// image the nearest edge pixel is used.
pub fn bilinear(img: &Image<f32>, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (x0, y0) = (x0 as isize, y0 as isize);
    let p = |dx: isize, dy: isize| img.get_border(x0 + dx, y0 + dy, Border::Clamp);
    (1.0 - fx) * (1.0 - fy) * p(0, 0) + fx * (1.0 - fy) * p(1, 0) + (1.0 - fx) * fy * p(0, 1) + fx * fy * p(1, 1)
}
