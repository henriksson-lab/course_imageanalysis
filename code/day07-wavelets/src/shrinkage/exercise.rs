// Coding exercises for "Denoising by wavelet shrinkage". Replace each `todo!()` with your code,
// then check it with `cargo test -p day07 shrinkage::` and look at `cargo run -p day07 -- shrinkage`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

// ---- From the 2D wavelet unit (given): filters, the 1D filter bank and the 2D pyramid. ----

pub const HAAR: [f32; 2] = [std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2];
pub const DB2: [f32; 4] = [0.482_962_9, 0.836_516_3, 0.224_143_87, -0.129_409_52];
pub const DB4: [f32; 8] = [0.230_377_81, 0.714_846_6, 0.630_880_8, -0.027_983_77, -0.187_034_81, 0.030_841_38, 0.032_883_01, -0.010_597_4];

/// The high-pass filter `g[n] = (−1)^n h[L−1−n]`.
pub fn highpass(h: &[f32]) -> Vec<f32> {
    let l = h.len();
    (0..l).map(|n| if n % 2 == 0 { h[l - 1 - n] } else { -h[l - 1 - n] }).collect()
}

/// One level of the 1D analysis filter bank, periodic: returns `a` followed by `d` in one vector.
pub fn analysis_1d(x: &[f32], h: &[f32]) -> Vec<f32> {
    let g = highpass(h);
    let n = x.len();
    let mut out = vec![0.0; n];
    for k in 0..n / 2 {
        for i in 0..h.len() {
            out[k] += h[i] * x[(2 * k + i) % n];
            out[n / 2 + k] += g[i] * x[(2 * k + i) % n];
        }
    }
    out
}

/// One level of the 1D synthesis filter bank, periodic: the inverse of `analysis_1d`.
pub fn synthesis_1d(c: &[f32], h: &[f32]) -> Vec<f32> {
    let g = highpass(h);
    let n = c.len();
    let mut x = vec![0.0; n];
    for k in 0..n / 2 {
        for i in 0..h.len() {
            x[(2 * k + i) % n] += h[i] * c[k] + g[i] * c[n / 2 + k];
        }
    }
    x
}

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
    todo!("C1a: hard")
}

/// C1b. Soft thresholding: 0 if `|w| ≤ t`, otherwise `w` moved towards 0 by `t`.
pub fn soft(w: f32, t: f32) -> f32 {
    todo!("C1b: soft")
}

/// C2a. The robust noise estimate from detail coefficients: `median(|w|) / 0.6745`
/// (slide "Estimating σn from the image"). For an even count, the median is the mean of the
/// two middle values.
///
/// Hint: sort the magnitudes with `a.sort_by(|p, q| p.total_cmp(q))`.
pub fn median_sigma(values: &[f32]) -> f32 {
    todo!("C2a: median_sigma")
}

/// C2b. The universal threshold `σ √(2 ln N)` for `N` coefficients.
pub fn universal_threshold(sigma: f32, n: usize) -> f32 {
    todo!("C2b: universal_threshold")
}

/// C3. Threshold every detail coefficient of a pyramid with `levels` levels and leave the LL
/// band (the top-left `width/2^levels × height/2^levels` corner) alone
/// (slide "Leave the approximation alone").
///
/// Hint: `c.map_xy(|x, y, w| ...)` gives a new image with a value for every position.
pub fn threshold_details(c: &Image<f32>, levels: usize, t: f32, use_soft: bool) -> Image<f32> {
    todo!("C3: threshold_details")
}

/// C4. Wavelet shrinkage: transform, threshold the details at `t`, transform back.
///
/// Hint: `dwt2d` (given above), then C3, then `idwt2d`.
pub fn denoise(img: &Image<f32>, h: &[f32], levels: usize, t: f32, use_soft: bool) -> Image<f32> {
    todo!("C4: denoise")
}

/// C5. Cycle spinning (slide "Artefacts and cycle spinning"): for every shift `(dx, dy)` with
/// `dx, dy` in `0..shifts`, shift the image (wrapping around), denoise, shift back; return the
/// average of the `shifts²` results.
///
/// Hint: the shifted image is `Image::from_fn(w, h, |x, y| img[((x + dx) % w, (y + dy) % h)])`;
/// the denoised pixel `(x, y)` of it belongs at `((x + dx) % w, (y + dy) % h)` of the result.
pub fn cycle_spin(img: &Image<f32>, h: &[f32], levels: usize, t: f32, use_soft: bool, shifts: usize) -> Image<f32> {
    todo!("C5: cycle_spin")
}
