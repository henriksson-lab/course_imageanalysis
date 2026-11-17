// Model solution for the coding exercises of "Reconstruction as a linear system".
// The matrix A is stored by rows: row i lists the pixels that ray i touches, with their
// weights, as (pixel index, weight). Images are vectors of pixels, row by row.

use crate::algebraic_recon::tomo::Row;

/// C1a. The forward projection b = A x: for every ray, the weighted sum of its pixels.
pub fn forward(a: &[Row], x: &[f32]) -> Vec<f32> {
    let mut b = vec![0.0; a.len()];
    for (i, row) in a.iter().enumerate() {
        for &(j, w) in row {
            b[i] += w * x[j];
        }
    }
    b
}

/// C1b. The back-projection Aᵀ y: every ray adds its value y_i, with the same weights, to its
/// pixels (slide "Aᵀ is back-projection").
pub fn back(a: &[Row], y: &[f32], n_pixels: usize) -> Vec<f32> {
    let mut x = vec![0.0; n_pixels];
    for (i, row) in a.iter().enumerate() {
        for &(j, w) in row {
            x[j] += w * y[i];
        }
    }
    x
}

/// C2. One sweep of ART (Kaczmarz) over all rays in order, changing `x` in place: for ray i,
/// error = b_i − a_i·x; add ω · error / ‖a_i‖² · a_i to x (slide "The math: Kaczmarz and
/// SIRT"). With weights 1 this spreads the error evenly over the pixels of the ray.
pub fn art_sweep(a: &[Row], b: &[f32], x: &mut [f32], omega: f32) {
    for (i, row) in a.iter().enumerate() {
        let mut sum = 0.0;
        let mut norm2 = 0.0;
        for &(j, w) in row {
            sum += w * x[j];
            norm2 += w * w;
        }
        if norm2 == 0.0 {
            continue; // a ray that misses the image
        }
        let step = omega * (b[i] - sum) / norm2;
        for &(j, w) in row {
            x[j] += step * w;
        }
    }
}

/// C3. One SIRT step, changing `x` in place: the error of every ray with the current x, divided
/// by the ray's weight sum, back-projected, and at every pixel divided by the weight sum of the
/// rays through it; times ω.   x ← x + ω W_c Aᵀ W_r (b − A x)
pub fn sirt_step(a: &[Row], b: &[f32], x: &mut [f32], omega: f32) {
    let current = forward(a, x);
    let mut correction = vec![0.0; x.len()];
    let mut column_sums = vec![0.0; x.len()];
    for (i, row) in a.iter().enumerate() {
        let row_sum: f32 = row.iter().map(|&(_, w)| w).sum();
        if row_sum == 0.0 {
            continue;
        }
        let r = (b[i] - current[i]) / row_sum;
        for &(j, w) in row {
            correction[j] += w * r;
            column_sums[j] += w;
        }
    }
    for j in 0..x.len() {
        if column_sums[j] > 0.0 {
            x[j] += omega * correction[j] / column_sums[j];
        }
    }
}
