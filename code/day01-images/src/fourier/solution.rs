// Model solution for the coding exercises of "Images in frequency space".

use imgcourse::{Complex, Image};
use std::f32::consts::PI;

/// C1. The discrete Fourier transform, straight from the definition:
/// `X[k] = sum over n of x[n] e^{-2πi kn/N}`. (Slow, `N²` steps; the FFT is fast.)
pub fn dft(x: &[Complex]) -> Vec<Complex> {
    let n = x.len();
    (0..n)
        .map(|k| {
            let mut sum = Complex::ZERO;
            for (j, &xj) in x.iter().enumerate() {
                sum += xj * Complex::from_angle(-2.0 * PI * (k * j % n) as f32 / n as f32);
            }
            sum
        })
        .collect()
}

/// C2. The inverse DFT: `x[n] = (1/N) sum over k of X[k] e^{+2πi kn/N}`.
pub fn idft(big_x: &[Complex]) -> Vec<Complex> {
    let n = big_x.len();
    (0..n)
        .map(|j| {
            let mut sum = Complex::ZERO;
            for (k, &xk) in big_x.iter().enumerate() {
                sum += xk * Complex::from_angle(2.0 * PI * (k * j % n) as f32 / n as f32);
            }
            sum / n as f32
        })
        .collect()
}

/// C3. The frequency response of a small 1D kernel `w` (odd length, centre in the middle)
/// on signals of length `n`: the number each wave `k = 0..n` is multiplied by.
pub fn frequency_response(w: &[f32], n: usize) -> Vec<Complex> {
    // As a signal of length n with the centre weight at index 0, so that filtering is
    // (circular) convolution with it. Convolution flips the kernel; the lecture's filters
    // do not, so weight j - half goes to index -(j - half), wrapped around.
    let half = (w.len() / 2) as isize;
    let mut signal = vec![Complex::ZERO; n];
    for (j, &wj) in w.iter().enumerate() {
        let offset = j as isize - half;
        signal[(-offset).rem_euclid(n as isize) as usize] += Complex::from(wj);
    }
    dft(&signal)
}

/// C4. The 2D DFT (or, with `inverse`, the inverse 2D DFT): the 1D transform of every row,
/// then of every column of the result.
pub fn dft2(img: &Image<Complex>, inverse: bool) -> Image<Complex> {
    let t = |v: &[Complex]| if inverse { idft(v) } else { dft(v) };
    let (w, h) = (img.width(), img.height());
    let mut rows = Image::new(w, h, Complex::ZERO);
    for y in 0..h {
        for (x, v) in t(img.row(y)).into_iter().enumerate() {
            rows[(x, y)] = v;
        }
    }
    let mut out = Image::new(w, h, Complex::ZERO);
    for x in 0..w {
        let column: Vec<Complex> = (0..h).map(|y| rows[(x, y)]).collect();
        for (y, v) in t(&column).into_iter().enumerate() {
            out[(x, y)] = v;
        }
    }
    out
}

/// C5. A low-pass filter in frequency space: transform, keep the coefficients whose
/// frequency `(u, v)` lies within `radius` of 0, transform back, keep the real part.
/// Index `k` above half the size is the negative frequency `k - N`.
pub fn lowpass(img: &Image<f32>, radius: f32) -> Image<f32> {
    let (w, h) = (img.width(), img.height());
    let freq = |k: usize, n: usize| if k <= n / 2 { k as f32 } else { k as f32 - n as f32 };
    let spectrum = dft2(&img.map(Complex::from), false);
    let kept = spectrum.map_xy(|kx, ky, c| {
        let (u, v) = (freq(kx, w), freq(ky, h));
        if u * u + v * v <= radius * radius { c } else { Complex::ZERO }
    });
    dft2(&kept, true).map(|c| c.re)
}
