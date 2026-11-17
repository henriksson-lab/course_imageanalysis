// Model solution for the coding exercises of "The 2D wavelet transform and sparsity of images".

use imgcourse::Image;

// ---- From the filter bank unit (given, in `src/common.rs`): orthogonal low-pass filters and
// one 1D level. ----

pub use crate::common::{analysis_1d, highpass, synthesis_1d, DB2, DB4, HAAR};

// ---- The exercises. ----

/// C1. One level of the separable 2D transform: `analysis_1d` on every row, then on every
/// column of the result (slide "Separable: rows, then columns"). The four quarter-size bands
/// sit in place: LL top left, V (high-pass along x) top right, H (high-pass along y) bottom
/// left, D bottom right. Width and height must be even.
pub fn dwt2d_level(img: &Image<f32>, h: &[f32]) -> Image<f32> {
    let (w, ht) = (img.width(), img.height());
    let mut rows = Image::new(w, ht, 0.0);
    for y in 0..ht {
        let r = analysis_1d(img.row(y), h);
        for x in 0..w {
            rows[(x, y)] = r[x];
        }
    }
    let mut out = Image::new(w, ht, 0.0);
    for x in 0..w {
        let column: Vec<f32> = (0..ht).map(|y| rows[(x, y)]).collect();
        let c = analysis_1d(&column, h);
        for y in 0..ht {
            out[(x, y)] = c[y];
        }
    }
    out
}

/// C2. The inverse of C1: `synthesis_1d` on every column, then on every row.
pub fn idwt2d_level(c: &Image<f32>, h: &[f32]) -> Image<f32> {
    let (w, ht) = (c.width(), c.height());
    let mut cols = Image::new(w, ht, 0.0);
    for x in 0..w {
        let column: Vec<f32> = (0..ht).map(|y| c[(x, y)]).collect();
        let s = synthesis_1d(&column, h);
        for y in 0..ht {
            cols[(x, y)] = s[y];
        }
    }
    let mut out = Image::new(w, ht, 0.0);
    for y in 0..ht {
        let s = synthesis_1d(cols.row(y), h);
        for x in 0..w {
            out[(x, y)] = s[x];
        }
    }
    out
}

/// C3a. The subband pyramid (Mallat layout): `levels` times C1, each time on the LL band in the
/// top-left corner, which halves in width and height every level (slide "The subband pyramid").
pub fn dwt2d(img: &Image<f32>, h: &[f32], levels: usize) -> Image<f32> {
    let mut c = img.clone();
    let (mut w, mut ht) = (img.width(), img.height());
    for _ in 0..levels {
        let band = dwt2d_level(&c.crop(0, 0, w, ht), h);
        for (x, y) in band.coords() {
            c[(x, y)] = band[(x, y)];
        }
        w /= 2;
        ht /= 2;
    }
    c
}

/// C3b. The inverse of C3a: C2 on the top-left corner, from the coarsest level out.
pub fn idwt2d(c: &Image<f32>, h: &[f32], levels: usize) -> Image<f32> {
    let mut x = c.clone();
    let (mut w, mut ht) = (c.width() >> levels, c.height() >> levels);
    for _ in 0..levels {
        w *= 2;
        ht *= 2;
        let part = idwt2d_level(&x.crop(0, 0, w, ht), h);
        for (i, j) in part.coords() {
            x[(i, j)] = part[(i, j)];
        }
    }
    x
}

/// C4a. Keep the given fraction (0 to 1) of all coefficients, the largest in magnitude; set the
/// others to 0. Returns the kept coefficients and the sum of the squares of the dropped ones.
pub fn keep_fraction(c: &Image<f32>, fraction: f32) -> (Image<f32>, f32) {
    let mut sizes: Vec<f32> = c.pixels().iter().map(|v| v.abs()).collect();
    sizes.sort_by(|a, b| b.total_cmp(a)); // largest first
    let k = (fraction * sizes.len() as f32).round() as usize;
    if k == 0 {
        return (c.map(|_| 0.0), c.pixels().iter().map(|v| v * v).sum());
    }
    let threshold = sizes[k - 1];
    let kept = c.map(|v| if v.abs() >= threshold { v } else { 0.0 });
    let mut dropped = 0.0;
    for (a, b) in c.pixels().iter().zip(kept.pixels()) {
        dropped += (a - b) * (a - b);
    }
    (kept, dropped)
}

/// C4b. The peak signal-to-noise ratio in dB: `10 log10(peak² / MSE)`, with the mean squared
/// difference between the two images (slide "The math: K-term approximation and PSNR").
pub fn psnr(a: &Image<f32>, b: &Image<f32>, peak: f32) -> f32 {
    let mut sum = 0.0f64;
    for (p, q) in a.pixels().iter().zip(b.pixels()) {
        sum += ((p - q) as f64).powi(2);
    }
    let mse = sum / a.pixels().len() as f64;
    (10.0 * (peak as f64 * peak as f64 / mse).log10()) as f32
}
