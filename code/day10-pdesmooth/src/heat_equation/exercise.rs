// Coding exercises for "Smoothing as heat flow". Replace each `todo!()` with your code,
// then check it with `cargo test -p day10 heat_equation::` and look at `cargo run -p day10 -- heat-equation`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};

/// C1. One explicit heat step on a row: every value moves towards its two neighbours,
/// `u[i] + dt * (u[i-1] + u[i+1] - 2 u[i])`. Mirror border: the value outside the row is a
/// copy of the end value, so no heat leaves and the total is kept.
///
/// Hint: write the result into a new `Vec`, so that every step reads only old values.
pub fn heat_step_1d(row: &[f32], dt: f32) -> Vec<f32> {
    todo!("C1: heat_step_1d")
}

/// C2. One explicit heat step on an image (slide "Average your neighbours, repeat"):
/// `u + dt * (sum of the 4 neighbours - 4 u)`, the neighbours outside read with `border`.
/// The result goes into a new image.
///
/// Hint: `img.get_border(x as isize - 1, y as isize, border)` is the left neighbour, also at
/// the edge of the image.
pub fn heat_step(img: &Image<f32>, dt: f32, border: Border) -> Image<f32> {
    todo!("C2: heat_step")
}

/// C3. Heat flow for `steps` explicit steps of size `dt` (time `t = steps * dt`).
pub fn heat_flow(img: &Image<f32>, dt: f32, steps: usize, border: Border) -> Image<f32> {
    todo!("C3: heat_flow")
}

/// C4. One implicit heat step on a row (slide "Implicit steps"): solve
/// `(I + dt L) u = row` for `u`, where `L` is the path Laplacian (every value has the
/// neighbours it has: the two end values only one, as with a mirror border).
/// The matrix is tridiagonal: eliminate downwards, then substitute back upwards.
///
/// Hint: row `i` of the matrix has `-dt` left and right of the diagonal, and
/// `1 + dt * (number of neighbours)` on it. Going down, rewrite every equation as
/// `u[i] + c[i] * u[i+1] = d[i]` using the one above it; then the last equation gives
/// `u[n-1] = d[n-1]`, and going up `u[i] = d[i] - c[i] * u[i+1]`.
pub fn implicit_step_1d(row: &[f32], dt: f32) -> Vec<f32> {
    todo!("C4: implicit_step_1d")
}

/// Not an exercise: a Gaussian blur (two passes of a sampled, normalised 1D Gaussian of
/// radius 4σ, mirror border), to compare heat flow with.
pub fn gaussian_blur(img: &Image<f32>, sigma: f32) -> Image<f32> {
    let r = (4.0 * sigma).ceil() as isize;
    let w: Vec<f32> = (-r..=r).map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let total: f32 = w.iter().sum();
    let pass = |src: &Image<f32>, along_x: bool| {
        src.map_xy(|x, y, _| {
            let mut sum = 0.0;
            for d in -r..=r {
                let (xx, yy) = if along_x { (x as isize + d, y as isize) } else { (x as isize, y as isize + d) };
                sum += w[(d + r) as usize] * src.get_border(xx, yy, Border::Mirror);
            }
            sum / total
        })
    };
    pass(&pass(img, true), false)
}
