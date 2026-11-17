// Model solution for the coding exercises of "Multiresolution, filter banks and Daubechies wavelets".

/// Low-pass filters `h` of orthogonal wavelets (the high-pass follows from them, C1); Haar,
/// db2 and db4 are in `src/common.rs`.
pub use crate::common::{DB2, DB4, HAAR};
pub const DB3: [f32; 6] = [0.332_670_55, 0.806_891_5, 0.459_877_5, -0.135_011_02, -0.085_441_27, 0.035_226_29];

/// C1. The high-pass filter of an orthogonal filter bank: the low-pass reversed, with every
/// second sign flipped, `g[n] = (−1)^n h[L−1−n]` (slide "The math: perfect reconstruction").
pub fn highpass(h: &[f32]) -> Vec<f32> {
    let l = h.len();
    (0..l).map(|n| if n % 2 == 0 { h[l - 1 - n] } else { -h[l - 1 - n] }).collect()
}

/// C2. The number of vanishing moments of a high-pass filter: how many of the sums
/// `sum over n of n^m g[n]`, for `m = 0, 1, 2, ...`, are zero (slide "The math: vanishing moments").
/// A sum counts as zero when it is below `1e-4` times the sum of the sizes of its terms.
pub fn vanishing_moments(g: &[f32]) -> usize {
    let mut m = 0;
    loop {
        let mut sum = 0.0f64;
        let mut size = 0.0f64;
        for (n, &gn) in g.iter().enumerate() {
            let term = (n as f64).powi(m as i32) * gn as f64;
            sum += term;
            size += term.abs();
        }
        if sum.abs() > 1e-4 * size || m == g.len() {
            return m;
        }
        m += 1;
    }
}

/// C3. One level of the analysis filter bank with a periodic border: filter with `h` and `g`
/// and keep every second sample, `a[k] = sum over n of h[n] x[(2k + n) mod N]`, `d[k]` the same
/// with `g` (slide "The DWT is a filter bank"). Returns `(a, d)`, each of length `N/2`.
pub fn analysis(x: &[f32], h: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let g = highpass(h);
    let n = x.len();
    let mut a = vec![0.0; n / 2];
    let mut d = vec![0.0; n / 2];
    for k in 0..n / 2 {
        for i in 0..h.len() {
            let xi = x[(2 * k + i) % n];
            a[k] += h[i] * xi;
            d[k] += g[i] * xi;
        }
    }
    (a, d)
}

/// C4. One level of the synthesis filter bank (periodic): upsample, filter and add. Every
/// coefficient puts its filter back where it came from: `x[(2k + n) mod N] += h[n] a[k] + g[n] d[k]`.
pub fn synthesis(a: &[f32], d: &[f32], h: &[f32]) -> Vec<f32> {
    let g = highpass(h);
    let n = 2 * a.len();
    let mut x = vec![0.0; n];
    for k in 0..a.len() {
        for i in 0..h.len() {
            x[(2 * k + i) % n] += h[i] * a[k] + g[i] * d[k];
        }
    }
    x
}

/// C5a. The cascade: `levels` analysis steps, each on the approximation of the previous one.
/// The result is `(a_L | d_L | ... | d_2 | d_1)`, coarse to fine, `N` numbers in all.
pub fn dwt(x: &[f32], h: &[f32], levels: usize) -> Vec<f32> {
    let mut c = x.to_vec();
    let mut len = x.len();
    for _ in 0..levels {
        let (a, d) = analysis(&c[..len], h);
        c[..len / 2].copy_from_slice(&a);
        c[len / 2..len].copy_from_slice(&d);
        len /= 2;
    }
    c
}

/// C5b. The inverse cascade: `levels` synthesis steps, from the coarsest level up.
pub fn idwt(c: &[f32], h: &[f32], levels: usize) -> Vec<f32> {
    let mut x = c.to_vec();
    let mut len = c.len() >> levels;
    for _ in 0..levels {
        let up = synthesis(&x[..len], &x[len..2 * len], h);
        x[..2 * len].copy_from_slice(&up);
        len *= 2;
    }
    x
}
