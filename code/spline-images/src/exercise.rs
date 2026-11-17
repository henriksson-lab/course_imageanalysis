// Coding exercises for "Splines on images: interpolation and free-form deformation". Replace
// each `todo!()` with your code, then check it with `cargo test -p spline-images` and look at
// `cargo run -p spline-images`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};

/// The pole of the cubic B-spline prefilter, `z1 = √3 − 2`.
pub const Z1: f32 = -0.267_949_2;

/// The cubic B-spline β³ (from the B-spline unit).
pub fn cubic_bspline(x: f32) -> f32 {
    let a = x.abs();
    if a <= 1.0 {
        2.0 / 3.0 - a * a + 0.5 * a * a * a
    } else if a < 2.0 {
        (2.0 - a).powi(3) / 6.0
    } else {
        0.0
    }
}

/// C1. The prefilter along one row: coefficients `c` with `(c[k−1] + 4 c[k] + c[k+1]) / 6 = f[k]`,
/// where `c` outside the row is read with `Border::Mirror` (slide "The math: tensor product and
/// prefilter"). Two recursive passes with the pole `z1`, after a gain of 6:
/// left to right `c⁺[k] = 6 f[k] + z1 c⁺[k−1]`, right to left `c[k] = z1 (c[k+1] − c⁺[k])`.
///
/// Hint: the two passes need start values. Left to right, `c⁺[0]` is the sum over `j = 0, 1, ..., 14`
/// of `z1^j · 6 f[−j]`, reading `f[−j]` with the mirror: `Border::Mirror.index(-j, n).unwrap()`
/// gives its position in the row. Right to left, the mirror makes `c[n] = c[n−1]`, which gives
/// `c[n−1] = z1 / (z1 − 1) · c⁺[n−1]`.
pub fn prefilter_1d(f: &[f32]) -> Vec<f32> {
    todo!("C1: prefilter_1d")
}

/// C2. The B-spline coefficients of an image: `prefilter_1d` along every row, then along every
/// column of the result.
///
/// Hint: `img.row(y)` is row `y` as a slice; collect a column with
/// `(0..h).map(|y| rows[(x, y)]).collect::<Vec<f32>>()`.
pub fn prefilter(img: &Image<f32>) -> Image<f32> {
    todo!("C2: prefilter")
}

/// C3. The value of the tensor-product cubic B-spline with coefficients `coef` at a position
/// `(x, y)` between pixels: `sum over k, l of coef[k, l] β³(x − k) β³(y − l)`, over the 4 × 4
/// coefficients near `(x, y)`; outside the image the coefficients are read with `Border::Mirror`.
///
/// Hint: `k` runs from `floor(x) − 1` to `floor(x) + 2` (as `isize`), `l` likewise;
/// `coef.get_border(k, l, Border::Mirror)` reads a coefficient.
pub fn bspline_value(coef: &Image<f32>, x: f32, y: f32) -> f32 {
    todo!("C3: bspline_value")
}

/// C4a. The cubic convolution kernel (a = −1/2): `1.5|s|³ − 2.5|s|² + 1` for `|s| ≤ 1`,
/// `−0.5|s|³ + 2.5|s|² − 4|s| + 2` for `1 < |s| < 2`, else 0.
pub fn cubic_convolution_kernel(s: f32) -> f32 {
    todo!("C4a: cubic_convolution_kernel")
}

/// C4b. Cubic convolution interpolation at `(x, y)`: like C3, but on the **pixels** themselves
/// (no prefilter) with the kernel of C4a; outside the image `Border::Mirror`.
pub fn cubic_convolution(img: &Image<f32>, x: f32, y: f32) -> f32 {
    todo!("C4b: cubic_convolution")
}

/// C5. The free-form deformation at pixel position `(x, y)`: `u = sum over k, l of
/// d[k, l] β³(x/h − k) β³(y/h − l)`, where control point `(k, l)` sits at `(k h, l h)` and has
/// the displacement `d[(k, l)]` (slide "The math: the FFD displacement field"). Control points
/// outside the grid count as 0.
///
/// Hint: the same 4 × 4 loop as C3, in grid units `x/h`, `y/h`; skip `(k, l)` for which
/// `d.contains(k, l)` is false.
pub fn ffd_displacement(d: &Image<[f32; 2]>, h: f32, x: f32, y: f32) -> [f32; 2] {
    todo!("C5: ffd_displacement")
}
