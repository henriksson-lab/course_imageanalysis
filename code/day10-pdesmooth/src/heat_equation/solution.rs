// Model solution for the coding exercises of "Smoothing as heat flow".

use imgcourse::{Border, Image};

/// C1. One explicit heat step on a row: every value moves towards its two neighbours,
/// `u[i] + dt * (u[i-1] + u[i+1] - 2 u[i])`. Mirror border: the value outside the row is a
/// copy of the end value, so no heat leaves and the total is kept.
pub fn heat_step_1d(row: &[f32], dt: f32) -> Vec<f32> {
    let n = row.len();
    let mut out = vec![0.0; n];
    for i in 0..n {
        let left = if i > 0 { row[i - 1] } else { row[i] };
        let right = if i + 1 < n { row[i + 1] } else { row[i] };
        out[i] = row[i] + dt * (left + right - 2.0 * row[i]);
    }
    out
}

/// C2. One explicit heat step on an image (slide "Average your neighbours, repeat"):
/// `u + dt * (sum of the 4 neighbours - 4 u)`, the neighbours outside read with `border`.
/// The result goes into a new image.
pub fn heat_step(img: &Image<f32>, dt: f32, border: Border) -> Image<f32> {
    let mut out = Image::new(img.width(), img.height(), 0.0);
    for (x, y) in img.coords() {
        let (xi, yi) = (x as isize, y as isize);
        let u = img[(x, y)];
        let neighbours = img.get_border(xi - 1, yi, border)
            + img.get_border(xi + 1, yi, border)
            + img.get_border(xi, yi - 1, border)
            + img.get_border(xi, yi + 1, border);
        out[(x, y)] = u + dt * (neighbours - 4.0 * u);
    }
    out
}

/// C3. Heat flow for `steps` explicit steps of size `dt` (time `t = steps * dt`).
pub fn heat_flow(img: &Image<f32>, dt: f32, steps: usize, border: Border) -> Image<f32> {
    let mut u = img.clone();
    for _ in 0..steps {
        u = heat_step(&u, dt, border);
    }
    u
}

/// C4. One implicit heat step on a row (slide "Implicit steps"): solve
/// `(I + dt L) u = row` for `u`, where `L` is the path Laplacian (every value has the
/// neighbours it has: the two end values only one, as with a mirror border).
/// The matrix is tridiagonal: eliminate downwards, then substitute back upwards.
pub fn implicit_step_1d(row: &[f32], dt: f32) -> Vec<f32> {
    let n = row.len();
    if n == 1 {
        return row.to_vec();
    }
    // Row i of the matrix: -dt on both sides, 1 + dt * (number of neighbours) on the diagonal.
    let off = -dt;
    let diag = |i: usize| 1.0 + dt * if i == 0 || i == n - 1 { 1.0 } else { 2.0 };
    // Downwards: after this, row i reads  u[i] + c[i] u[i+1] = d[i].
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];
    c[0] = off / diag(0);
    d[0] = row[0] / diag(0);
    for i in 1..n {
        let m = diag(i) - off * c[i - 1];
        c[i] = off / m;
        d[i] = (row[i] - off * d[i - 1]) / m;
    }
    // Upwards: the last value is known, then each one from the next.
    let mut u = vec![0.0; n];
    u[n - 1] = d[n - 1];
    for i in (0..n - 1).rev() {
        u[i] = d[i] - c[i] * u[i + 1];
    }
    u
}

/// Not an exercise: a Gaussian blur (radius 4σ, mirror border), in `src/common.rs`.
pub use crate::common::gaussian_blur;
