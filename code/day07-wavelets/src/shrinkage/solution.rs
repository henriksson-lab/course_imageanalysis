// Model solution for the coding exercises of "Denoising by wavelet shrinkage".

use imgcourse::Image;

// ---- From the 2D wavelet unit (given): filters and the 1D filter bank (in `src/common.rs`),
// and the 2D pyramid. ----

pub use crate::common::{analysis_1d, highpass, synthesis_1d, DB2, DB4, HAAR};

/// One level of the separable 2D transform: `analysis_1d` on every row, then on every
/// column of the result. The four quarter-size bands
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

/// The inverse of `dwt2d_level`: `synthesis_1d` on every column, then on every row.
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

/// The subband pyramid (Mallat layout): `levels` times `dwt2d_level`, each time on the LL
/// band in the top-left corner, which halves in width and height every level.
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

/// The inverse of `dwt2d`: `idwt2d_level` on the top-left corner, from the coarsest level out.
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

// ---- The exercises. ----

/// C1a. Hard thresholding: keep `w` if `|w| > t`, else 0 (slide "Hard and soft thresholding").
pub fn hard(w: f32, t: f32) -> f32 {
    if w.abs() > t {
        w
    } else {
        0.0
    }
}

/// C1b. Soft thresholding: 0 if `|w| ≤ t`, otherwise `w` moved towards 0 by `t`.
pub fn soft(w: f32, t: f32) -> f32 {
    if w > t {
        w - t
    } else if w < -t {
        w + t
    } else {
        0.0
    }
}

/// C2a. The robust noise estimate from detail coefficients: `median(|w|) / 0.6745`
/// (slide "Estimating σn from the image"). For an even count, the median is the mean of the
/// two middle values.
pub fn median_sigma(values: &[f32]) -> f32 {
    let mut a: Vec<f32> = values.iter().map(|v| v.abs()).collect();
    a.sort_by(|p, q| p.total_cmp(q));
    let n = a.len();
    let median = if n % 2 == 1 { a[n / 2] } else { 0.5 * (a[n / 2 - 1] + a[n / 2]) };
    median / 0.6745
}

/// C2b. The universal threshold `σ √(2 ln N)` for `N` coefficients.
pub fn universal_threshold(sigma: f32, n: usize) -> f32 {
    sigma * (2.0 * (n as f32).ln()).sqrt()
}

/// C3. Threshold every detail coefficient of a pyramid with `levels` levels and leave the LL
/// band (the top-left `width/2^levels × height/2^levels` corner) alone
/// (slide "Leave the approximation alone").
pub fn threshold_details(c: &Image<f32>, levels: usize, t: f32, use_soft: bool) -> Image<f32> {
    let (llw, llh) = (c.width() >> levels, c.height() >> levels);
    c.map_xy(|x, y, w| {
        if x < llw && y < llh {
            w
        } else if use_soft {
            soft(w, t)
        } else {
            hard(w, t)
        }
    })
}

/// C4. Wavelet shrinkage: transform, threshold the details at `t`, transform back.
pub fn denoise(img: &Image<f32>, h: &[f32], levels: usize, t: f32, use_soft: bool) -> Image<f32> {
    let c = dwt2d(img, h, levels);
    idwt2d(&threshold_details(&c, levels, t, use_soft), h, levels)
}

/// C5. Cycle spinning (slide "Artefacts and cycle spinning"): for every shift `(dx, dy)` with
/// `dx, dy` in `0..shifts`, shift the image (wrapping around), denoise, shift back; return the
/// average of the `shifts²` results.
pub fn cycle_spin(img: &Image<f32>, h: &[f32], levels: usize, t: f32, use_soft: bool, shifts: usize) -> Image<f32> {
    let (w, ht) = (img.width(), img.height());
    let mut sum = Image::new(w, ht, 0.0);
    for dy in 0..shifts {
        for dx in 0..shifts {
            let shifted = Image::from_fn(w, ht, |x, y| img[((x + dx) % w, (y + dy) % ht)]);
            let d = denoise(&shifted, h, levels, t, use_soft);
            for (x, y) in img.coords() {
                sum[((x + dx) % w, (y + dy) % ht)] += d[(x, y)];
            }
        }
    }
    let n = (shifts * shifts) as f32;
    sum.map(|v| v / n)
}
