// Coding exercises for "Multiresolution, filter banks and Daubechies wavelets". Replace each
// `todo!()` with your code, then check it with `cargo test -p filterbank` and look at
// `cargo run -p filterbank`.
#![allow(unused_variables, unused_imports)]

/// Low-pass filters `h` of orthogonal wavelets (the high-pass follows from them, C1).
pub const HAAR: [f32; 2] = [std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2];
pub const DB2: [f32; 4] = [0.482_962_9, 0.836_516_3, 0.224_143_87, -0.129_409_52];
pub const DB3: [f32; 6] = [0.332_670_55, 0.806_891_5, 0.459_877_5, -0.135_011_02, -0.085_441_27, 0.035_226_29];
pub const DB4: [f32; 8] = [0.230_377_81, 0.714_846_6, 0.630_880_8, -0.027_983_77, -0.187_034_81, 0.030_841_38, 0.032_883_01, -0.010_597_4];

/// C1. The high-pass filter of an orthogonal filter bank: the low-pass reversed, with every
/// second sign flipped, `g[n] = (−1)^n h[L−1−n]` (slide "The math: perfect reconstruction").
pub fn highpass(h: &[f32]) -> Vec<f32> {
    todo!("C1: highpass")
}

/// C2. The number of vanishing moments of a high-pass filter: how many of the sums
/// `sum over n of n^m g[n]`, for `m = 0, 1, 2, ...`, are zero (slide "The math: vanishing moments").
/// A sum counts as zero when it is below `1e-4` times the sum of the sizes of its terms.
///
/// Hint: compute in `f64` (`(n as f64).powi(m)`); count up `m` until a sum is not zero.
pub fn vanishing_moments(g: &[f32]) -> usize {
    todo!("C2: vanishing_moments")
}

/// C3. One level of the analysis filter bank with a periodic border: filter with `h` and `g`
/// and keep every second sample, `a[k] = sum over n of h[n] x[(2k + n) mod N]`, `d[k]` the same
/// with `g` (slide "The DWT is a filter bank"). Returns `(a, d)`, each of length `N/2`.
pub fn analysis(x: &[f32], h: &[f32]) -> (Vec<f32>, Vec<f32>) {
    todo!("C3: analysis")
}

/// C4. One level of the synthesis filter bank (periodic): upsample, filter and add. Every
/// coefficient puts its filter back where it came from: `x[(2k + n) mod N] += h[n] a[k] + g[n] d[k]`.
pub fn synthesis(a: &[f32], d: &[f32], h: &[f32]) -> Vec<f32> {
    todo!("C4: synthesis")
}

/// C5a. The cascade: `levels` analysis steps, each on the approximation of the previous one.
/// The result is `(a_L | d_L | ... | d_2 | d_1)`, coarse to fine, `N` numbers in all.
///
/// Hint: as the Haar transform of the last unit: keep a length `len`, replace `c[..len]` by
/// `a` followed by `d`, halve `len`.
pub fn dwt(x: &[f32], h: &[f32], levels: usize) -> Vec<f32> {
    todo!("C5a: dwt")
}

/// C5b. The inverse cascade: `levels` synthesis steps, from the coarsest level up.
///
/// Hint: the coarsest approximation has length `N >> levels` (`N / 2^levels`).
pub fn idwt(c: &[f32], h: &[f32], levels: usize) -> Vec<f32> {
    todo!("C5b: idwt")
}
