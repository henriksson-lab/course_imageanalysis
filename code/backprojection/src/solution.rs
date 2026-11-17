// Model solution for the coding exercises of "Back-projection and the Fourier slice theorem".
// Coordinates as in the lecture: origin at the image centre, y up, θ counter-clockwise
// (see `tomo.rs`: `centre`, `projection`, `sinogram`, `angles`).

use crate::tomo::centre;
use imgcourse::{Complex, Image};
use std::f32::consts::PI;

/// C1. Back-projection of two projections by hand (slide "By hand: three points, two angles"):
/// every pixel gets its column's value from `p0` plus its row's value from `p90`.
/// Bin k of `p0` is column k (x grows to the right); bin k of `p90` is the row at height
/// y = k − c, so bin 0 is the **bottom** row.
pub fn backproject_two(p0: &[f32], p90: &[f32]) -> Image<f32> {
    let n = p0.len();
    Image::from_fn(n, n, |px, py| p0[px] + p90[n - 1 - py])
}

/// C2. Back-projection of a sinogram (`sino[(a, k)]` = bin k at `thetas[a]`) onto an n × n
/// image: for every pixel (x, y) and every angle, read the sinogram at ρ = x cos θ + y sin θ
/// (between bins: linear interpolation, 0 outside the detector); add over the angles and
/// multiply by π / N_θ (the step dθ of the integral).
pub fn backproject(sino: &Image<f32>, thetas: &[f32], n: usize) -> Image<f32> {
    let c = centre(n);
    let bins = sino.height();
    let cb = centre(bins);
    let mut out = Image::new(n, n, 0.0);
    for (px, py) in out.coords() {
        let (x, y) = (px as f32 - c, c - py as f32);
        let mut sum = 0.0;
        for (a, &theta) in thetas.iter().enumerate() {
            let t = x * theta.cos() + y * theta.sin() + cb; // the bin position, between bins
            let k = t.floor();
            let f = t - k;
            let read = |k: f32| if k >= 0.0 && (k as usize) < bins { sino[(a, k as usize)] } else { 0.0 };
            sum += (1.0 - f) * read(k) + f * read(k + 1.0);
        }
        out[(px, py)] = sum * PI / thetas.len() as f32;
    }
    out
}

/// C3a. The size of the 1D DFT of a projection, in centred order: element i is frequency
/// ω = i − n/2, so ω = 0 is in the middle (element n/2).
pub fn projection_spectrum(p: &[f32]) -> Vec<f32> {
    let n = p.len();
    let spectrum = imgcourse::fft::fft(&p.iter().map(|&v| Complex::from(v)).collect::<Vec<_>>(), false);
    (0..n).map(|i| spectrum[(i + n - n / 2) % n].abs()).collect()
}

/// C3b. The size of the 2D DFT of an n × n image along the line through frequency 0 at angle θ,
/// in the same centred order: element i is the point ω (cos θ, sin θ) with ω = i − n/2, read
/// between frequencies by bilinear interpolation. In the DFT, index (u, v) is (x frequency,
/// row frequency), and rows go **down**: the point ω (cos θ, sin θ) is u = ω cos θ,
/// v = −ω sin θ. Frequencies wrap around: index −1 is n − 1.
pub fn spectrum_slice(spectrum: &Image<Complex>, theta: f32) -> Vec<f32> {
    let n = spectrum.width();
    let size = |u: isize, v: isize| spectrum[(u.rem_euclid(n as isize) as usize, v.rem_euclid(n as isize) as usize)].abs();
    (0..n)
        .map(|i| {
            let w = i as f32 - (n / 2) as f32;
            let (u, v) = (w * theta.cos(), -w * theta.sin());
            let (u0, v0) = (u.floor(), v.floor());
            let (fu, fv) = (u - u0, v - v0);
            let (u0, v0) = (u0 as isize, v0 as isize);
            (1.0 - fu) * (1.0 - fv) * size(u0, v0) + fu * (1.0 - fv) * size(u0 + 1, v0) + (1.0 - fu) * fv * size(u0, v0 + 1) + fu * fv * size(u0 + 1, v0 + 1)
        })
        .collect()
}
