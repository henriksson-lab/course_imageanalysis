//! A fast Fourier transform, for the days after the Fourier unit (where you wrote the plain
//! DFT yourself). Same numbers as the DFT: `X[k] = sum over n of x[n] e^{-2πi kn/N}`; the
//! inverse divides by `N`. Fast (`N log N`) when the length is a power of two; other lengths
//! fall back to the plain DFT, which is correct but slow for long signals.

use crate::{Complex, Image};
use std::f32::consts::PI;

/// The DFT of `x` (or, with `inverse`, the inverse DFT, including the factor `1/N`).
pub fn fft(x: &[Complex], inverse: bool) -> Vec<Complex> {
    let n = x.len();
    let mut out = if n.is_power_of_two() { radix2(x, inverse) } else { plain_dft(x, inverse) };
    if inverse {
        for v in &mut out {
            *v = *v / n as f32;
        }
    }
    out
}

/// The 2D DFT of an image (or, with `inverse`, the inverse): every row, then every column.
pub fn fft2(img: &Image<Complex>, inverse: bool) -> Image<Complex> {
    let (w, h) = (img.width(), img.height());
    let mut rows = Image::new(w, h, Complex::ZERO);
    for y in 0..h {
        for (x, v) in fft(img.row(y), inverse).into_iter().enumerate() {
            rows[(x, y)] = v;
        }
    }
    let mut out = Image::new(w, h, Complex::ZERO);
    for x in 0..w {
        let col: Vec<Complex> = (0..h).map(|y| rows[(x, y)]).collect();
        for (y, v) in fft(&col, inverse).into_iter().enumerate() {
            out[(x, y)] = v;
        }
    }
    out
}

/// The 2D DFT of a real image.
pub fn fft2_real(img: &Image<f32>) -> Image<Complex> {
    fft2(&img.map(Complex::from), false)
}

/// The frequency of index `k` in a transform of length `n`: `k` up to `n/2`, then `k - n`
/// (negative frequencies).
pub fn frequency(k: usize, n: usize) -> f32 {
    if k <= n / 2 { k as f32 } else { k as f32 - n as f32 }
}

/// The spectrum as a picture: `log(1 + |X|)`, with frequency 0 moved to the centre.
pub fn spectrum_picture(spectrum: &Image<Complex>) -> Image<f32> {
    let (w, h) = (spectrum.width(), spectrum.height());
    Image::from_fn(w, h, |x, y| (1.0 + spectrum[((x + w / 2) % w, (y + h / 2) % h)].abs()).ln())
}

fn plain_dft(x: &[Complex], inverse: bool) -> Vec<Complex> {
    let n = x.len();
    let sign = if inverse { 1.0 } else { -1.0 };
    (0..n)
        .map(|k| {
            let mut s = Complex::ZERO;
            for (j, &v) in x.iter().enumerate() {
                s += v * Complex::from_angle(sign * 2.0 * PI * ((k * j) % n) as f32 / n as f32);
            }
            s
        })
        .collect()
}

// Iterative radix-2 Cooley–Tukey: reorder by bit reversal, then combine pairs of half-length
// transforms ("butterflies") of length 2, 4, 8, ...
fn radix2(x: &[Complex], inverse: bool) -> Vec<Complex> {
    let n = x.len();
    let mut a = x.to_vec();
    if n <= 1 {
        return a;
    }
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j > i {
            a.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let step = Complex::from_angle(sign * 2.0 * PI / len as f32);
        for start in (0..n).step_by(len) {
            let mut w = Complex::from(1.0);
            for i in 0..len / 2 {
                let u = a[start + i];
                let v = a[start + i + len / 2] * w;
                a[start + i] = u + v;
                a[start + i + len / 2] = u - v;
                w = w * step;
            }
        }
        len *= 2;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_equals_plain_and_inverts() {
        for n in [1usize, 2, 8, 64, 12] {
            let x: Vec<Complex> = (0..n).map(|i| Complex::new((i as f32 * 0.37).sin(), (i % 3) as f32)).collect();
            let a = fft(&x, false);
            let b = plain_dft(&x, false);
            for (p, q) in a.iter().zip(&b) {
                assert!((*p - *q).abs() < 1e-3 * n as f32, "n = {n}");
            }
            for (p, q) in fft(&a, true).iter().zip(&x) {
                assert!((*p - *q).abs() < 1e-4, "n = {n}");
            }
        }
    }
}
