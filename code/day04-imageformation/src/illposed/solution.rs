// Model solution for the coding exercises of "Running the model backwards: an ill-posed problem".

use imgcourse::fft::{fft2, fft2_real};
use imgcourse::{Complex, Image};
use std::f32::consts::PI;

/// C1. Blur a periodic row (the right end continues at the left) with a kernel of odd length
/// whose middle element sits on the pixel: `g[i] = sum over j of kernel[j] · f[i + j − r]`.
pub fn circular_blur(f: &[f32], kernel: &[f32]) -> Vec<f32> {
    let n = f.len() as isize;
    let r = (kernel.len() / 2) as isize;
    (0..n)
        .map(|i| {
            let mut sum = 0.0;
            for (j, &k) in kernel.iter().enumerate() {
                sum += k * f[(i + j as isize - r).rem_euclid(n) as usize]; // wrap around
            }
            sum
        })
        .collect()
}

/// C2. The gains `H(k)` of a symmetric kernel on a periodic row of `n` pixels, for the waves
/// `k = 0, 1, ..., n − 1` (k periods along the row): `H(k) = sum over j of kernel[j] ·
/// cos(2π k (j − r) / n)`. These are the eigenvalues of the circulant blur matrix.
pub fn transfer(kernel: &[f32], n: usize) -> Vec<f32> {
    let r = (kernel.len() / 2) as f32;
    (0..n)
        .map(|k| kernel.iter().enumerate().map(|(j, &h)| h * (2.0 * PI * k as f32 * (j as f32 - r) / n as f32).cos()).sum())
        .collect()
}

/// C3. The condition number of the blur on `n` pixels: the largest `|H(k)|` divided by the
/// smallest. Infinite (`f32::INFINITY`) when some `|H(k)|` is zero (below `1e-6`, rounding).
pub fn condition_number(kernel: &[f32], n: usize) -> f32 {
    let h = transfer(kernel, n);
    let largest = h.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let smallest = h.iter().fold(f32::INFINITY, |m, v| m.min(v.abs()));
    if smallest < 1e-6 { f32::INFINITY } else { largest / smallest }
}

/// C4. The OTF of a PSF for a periodic `w × h` image: put the PSF into a `w × h` image of
/// zeros with its centre at pixel (0, 0), wrapping around the edges (the pixel that is
/// `(dx, dy)` from the PSF's centre goes to `(dx mod w, dy mod h)`), then take the 2D DFT.
pub fn otf2(psf: &Image<f32>, w: usize, h: usize) -> Image<Complex> {
    let (rx, ry) = ((psf.width() / 2) as isize, (psf.height() / 2) as isize);
    let mut big = Image::new(w, h, Complex::ZERO);
    for (i, j) in psf.coords() {
        let x = (i as isize - rx).rem_euclid(w as isize) as usize;
        let y = (j as isize - ry).rem_euclid(h as isize) as usize;
        big[(x, y)] += Complex::from(psf[(i, j)]);
    }
    fft2(&big, false)
}

/// C5. The inverse filter: transform the image, divide every frequency by the OTF, transform
/// back and keep the real part. Where `|H|` is zero (below `1e-6`, rounding) there is nothing
/// to divide by: that frequency is set to 0.
pub fn inverse_filter(g: &Image<f32>, otf: &Image<Complex>) -> Image<f32> {
    let spectrum = fft2_real(g);
    let restored = spectrum.zip_map(otf, |gk, hk| {
        if hk.abs() < 1e-6 {
            Complex::ZERO
        } else {
            // G / H = G · conj(H) / |H|²
            gk * hk.conj() / hk.norm_sqr()
        }
    });
    fft2(&restored, true).map(|z| z.re)
}

/// The forward model of a periodic image: blur by multiplying the spectrum with the OTF.
/// Given (not an exercise).
pub fn blur_periodic(f: &Image<f32>, otf: &Image<Complex>) -> Image<f32> {
    let spectrum = fft2_real(f).zip_map(otf, |a, b| a * b);
    fft2(&spectrum, true).map(|z| z.re)
}
