// Coding exercises for "Images in frequency space". Replace each `todo!()` with your code,
// then check it with `cargo test -p day01 fourier::` and look at `cargo run -p day01 -- fourier`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Complex, Image};
use std::f32::consts::PI;

/// C1. The discrete Fourier transform, straight from the definition:
/// `X[k] = sum over n of x[n] e^{-2πi kn/N}`. (Slow, `N²` steps; the FFT is fast.)
///
/// Hint: `Complex::from_angle(theta)` is `e^{iθ}`; complex numbers add and multiply
/// with `+` and `*`. Two loops: over `k` (the output) and over `n` (the sum).
pub fn dft(x: &[Complex]) -> Vec<Complex> {
    todo!("C1: dft")
}

/// C2. The inverse DFT: `x[n] = (1/N) sum over k of X[k] e^{+2πi kn/N}`.
pub fn idft(big_x: &[Complex]) -> Vec<Complex> {
    todo!("C2: idft")
}

/// C3. The frequency response of a small 1D kernel `w` (odd length, centre in the middle)
/// on signals of length `n`: the number each wave `k = 0..n` is multiplied by.
///
/// Hint: make a signal of length `n` that is 0 except for the weights, with the centre
/// weight at index 0 and the weight for offset `d` at index `-d` (wrapped around, so
/// offset +1 goes to index n − 1). Its DFT is the answer.
pub fn frequency_response(w: &[f32], n: usize) -> Vec<Complex> {
    todo!("C3: frequency_response")
}

/// C4. The 2D DFT (or, with `inverse`, the inverse 2D DFT): the 1D transform of every row,
/// then of every column of the result.
///
/// Hint: `img.row(y)` gives a row as a slice; build each column as a `Vec`.
pub fn dft2(img: &Image<Complex>, inverse: bool) -> Image<Complex> {
    todo!("C4: dft2")
}

/// C5. A low-pass filter in frequency space: transform, keep the coefficients whose
/// frequency `(u, v)` lies within `radius` of 0, transform back, keep the real part.
/// Index `k` above half the size is the negative frequency `k - N`.
///
/// Hint: `img.map(Complex::from)` turns grey values into complex numbers, `c.re` back.
pub fn lowpass(img: &Image<f32>, radius: f32) -> Image<f32> {
    todo!("C5: lowpass")
}
