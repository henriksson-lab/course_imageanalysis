// Model solution for the coding exercises of "Transform coding, the DCT and the KLT".

use imgcourse::Image;
use std::f32::consts::PI;

/// C1. The N × N DCT-II matrix, row by row: `d[k][n] = α_k cos(π (2n + 1) k / 2N)` with
/// α_0 = √(1/N) and α_k = √(2/N) for k ≥ 1 (slide "The math: the DCT-II"). Row k is the
/// k-th basis vector.
pub fn dct_matrix(n: usize) -> Vec<Vec<f32>> {
    let mut d = vec![vec![0.0; n]; n];
    for k in 0..n {
        let alpha = if k == 0 { (1.0 / n as f32).sqrt() } else { (2.0 / n as f32).sqrt() };
        for i in 0..n {
            d[k][i] = alpha * (PI * (2 * i + 1) as f32 * k as f32 / (2 * n) as f32).cos();
        }
    }
    d
}

/// C2a. The DCT of a signal: c = D x (one coefficient per row of D).
pub fn dct(x: &[f32]) -> Vec<f32> {
    let d = dct_matrix(x.len());
    let mut c = vec![0.0; x.len()];
    for k in 0..x.len() {
        for i in 0..x.len() {
            c[k] += d[k][i] * x[i];
        }
    }
    c
}

/// C2b. The inverse DCT: x = Dᵀ c, the signal as a sum of basis vectors weighted by the
/// coefficients.
pub fn idct(c: &[f32]) -> Vec<f32> {
    let d = dct_matrix(c.len());
    let mut x = vec![0.0; c.len()];
    for k in 0..c.len() {
        for i in 0..c.len() {
            x[i] += d[k][i] * c[k];
        }
    }
    x
}

/// C3a. The 2D DCT of a square block, C = D X Dᵀ: the 1D DCT of every row, then of every
/// column of the result. `out[(u, v)]` is the coefficient of horizontal frequency u and
/// vertical frequency v, so (0, 0) is the DC and the top row (v = 0) holds the horizontal
/// frequencies.
pub fn dct2(block: &Image<f32>) -> Image<f32> {
    separable(block, dct)
}

/// C3b. The inverse 2D DCT: the inverse 1D DCT of every row, then of every column.
pub fn idct2(coeffs: &Image<f32>) -> Image<f32> {
    separable(coeffs, idct)
}

/// Apply a 1D transform to every row, then to every column.
fn separable(img: &Image<f32>, transform: fn(&[f32]) -> Vec<f32>) -> Image<f32> {
    let (w, h) = (img.width(), img.height());
    let mut rows = Image::new(w, h, 0.0);
    for y in 0..h {
        for (x, v) in transform(img.row(y)).into_iter().enumerate() {
            rows[(x, y)] = v;
        }
    }
    let mut out = Image::new(w, h, 0.0);
    for x in 0..w {
        let column: Vec<f32> = (0..h).map(|y| rows[(x, y)]).collect();
        for (y, v) in transform(&column).into_iter().enumerate() {
            out[(x, y)] = v;
        }
    }
    out
}

/// C4. Block transform coding without quantisation: cut the image into 8 × 8 blocks, take the
/// DCT of every block, keep only the `fraction` of all coefficients (over all blocks) with the
/// largest size, set the rest to 0, and transform back. Returns the rebuilt image.
/// The image size must be a multiple of 8.
pub fn keep_largest_blocks(img: &Image<f32>, fraction: f32) -> Image<f32> {
    let (w, h) = (img.width(), img.height());
    // All coefficients, block by block, in one image of the same size.
    let mut coeffs = Image::new(w, h, 0.0);
    for by in (0..h).step_by(8) {
        for bx in (0..w).step_by(8) {
            let c = dct2(&img.crop(bx, by, 8, 8));
            for (x, y) in c.coords() {
                coeffs[(bx + x, by + y)] = c[(x, y)];
            }
        }
    }
    // The threshold: the size of the k-th largest coefficient.
    let mut sizes: Vec<f32> = coeffs.pixels().iter().map(|c| c.abs()).collect();
    sizes.sort_by(|a, b| b.total_cmp(a));
    let k = ((fraction * sizes.len() as f32).round() as usize).clamp(1, sizes.len());
    let threshold = sizes[k - 1];
    let kept = coeffs.map(|c| if c.abs() >= threshold { c } else { 0.0 });
    // Transform every block back.
    let mut out = Image::new(w, h, 0.0);
    for by in (0..h).step_by(8) {
        for bx in (0..w).step_by(8) {
            let block = idct2(&kept.crop(bx, by, 8, 8));
            for (x, y) in block.coords() {
                out[(bx + x, by + y)] = block[(x, y)];
            }
        }
    }
    out
}
