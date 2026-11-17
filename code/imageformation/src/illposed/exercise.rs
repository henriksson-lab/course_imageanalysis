// Coding exercises for "Running the model backwards: an ill-posed problem". Replace each `todo!()`
// with your code, then check it with `cargo test -p imageformation illposed::` and look at `cargo run -p imageformation -- illposed`.
#![allow(unused_variables, unused_imports)]

use imgcourse::fft::{fft2, fft2_real};
use imgcourse::{Complex, Image};
use std::f32::consts::PI;

/// C1. Blur a periodic row (the right end continues at the left) with a kernel of odd length
/// whose middle element sits on the pixel: `g[i] = sum over j of kernel[j] · f[i + j − r]`.
///
/// Hint: `(i + j − r).rem_euclid(n)` wraps a position into `0..n` (use `isize`).
pub fn circular_blur(f: &[f32], kernel: &[f32]) -> Vec<f32> {
    todo!("C1: circular_blur")
}

/// C2. The gains `H(k)` of a symmetric kernel on a periodic row of `n` pixels, for the waves
/// `k = 0, 1, ..., n − 1` (k periods along the row): `H(k) = sum over j of kernel[j] ·
/// cos(2π k (j − r) / n)`. These are the eigenvalues of the circulant blur matrix.
///
/// Hint: one sum per `k`; `r = kernel.len() / 2`.
pub fn transfer(kernel: &[f32], n: usize) -> Vec<f32> {
    todo!("C2: transfer")
}

/// C3. The condition number of the blur on `n` pixels: the largest `|H(k)|` divided by the
/// smallest. Infinite (`f32::INFINITY`) when some `|H(k)|` is zero (below `1e-6`, rounding).
///
/// Hint: use `transfer`, then the largest and the smallest of the absolute values.
pub fn condition_number(kernel: &[f32], n: usize) -> f32 {
    todo!("C3: condition_number")
}

/// C4. The OTF of a PSF for a periodic `w × h` image: put the PSF into a `w × h` image of
/// zeros with its centre at pixel (0, 0), wrapping around the edges (the pixel that is
/// `(dx, dy)` from the PSF's centre goes to `(dx mod w, dy mod h)`), then take the 2D DFT.
///
/// Hint: `Image::new(w, h, Complex::ZERO)`, then for every PSF pixel `(i, j)`:
/// `x = (i − rx).rem_euclid(w)`, `y = (j − ry).rem_euclid(h)`. Then `fft2(&big, false)`.
pub fn otf2(psf: &Image<f32>, w: usize, h: usize) -> Image<Complex> {
    todo!("C4: otf2")
}

/// C5. The inverse filter: transform the image, divide every frequency by the OTF, transform
/// back and keep the real part. Where `|H|` is zero (below `1e-6`, rounding) there is nothing
/// to divide by: that frequency is set to 0.
///
/// Hint: `fft2_real(g)`, then `zip_map` with the OTF: `G / H` is
/// `g * h.conj() / h.norm_sqr()`. Back with `fft2(&..., true)` and `.map(|z| z.re)`.
pub fn inverse_filter(g: &Image<f32>, otf: &Image<Complex>) -> Image<f32> {
    todo!("C5: inverse_filter")
}

/// The forward model of a periodic image: blur by multiplying the spectrum with the OTF.
/// Given (not an exercise).
pub fn blur_periodic(f: &Image<f32>, otf: &Image<Complex>) -> Image<f32> {
    let spectrum = fft2_real(f).zip_map(otf, |a, b| a * b);
    fft2(&spectrum, true).map(|z| z.re)
}
