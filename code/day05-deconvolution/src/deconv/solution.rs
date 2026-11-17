// Model solution for the coding exercises of "Linear deconvolution: inverse, Wiener, Tikhonov".

use imgcourse::fft::{fft2, fft2_real, frequency};
use imgcourse::Image;
use std::f32::consts::PI;

/// C1a. The gain of the truncated inverse at a frequency where the OTF is `h` (real):
/// `1/h` where `|h| > eps`, else 0.
pub fn truncated_gain(h: f32, eps: f32) -> f32 {
    if h.abs() > eps { 1.0 / h } else { 0.0 }
}

/// C1b. The gain of the Wiener filter, real OTF `h`, signal-to-noise ratio `snr`:
/// `h / (h² + 1/snr)`.
pub fn wiener_gain(h: f32, snr: f32) -> f32 {
    h / (h * h + 1.0 / snr)
}

/// C1c. The gain of Tikhonov regularisation, real OTF `h`, weight `lambda`, and `d2` the
/// squared size of the penalty at this frequency (1 for the energy penalty, `L(f)²` for the
/// roughness penalty): `h / (h² + λ d2)`.
pub fn tikhonov_gain(h: f32, lambda: f32, d2: f32) -> f32 {
    h / (h * h + lambda * d2)
}

/// C2a. The OTF of a Gaussian PSF with standard deviation `sigma` pixels for a periodic
/// `w × h` image, in the layout of the 2D DFT: at pixel `(x, y)` the frequency is
/// `fx = frequency(x, w) / w`, `fy = frequency(y, h) / h` cycles per pixel, and
/// `H = exp(−2π² σ² (fx² + fy²))`.
pub fn gaussian_otf(sigma: f32, w: usize, h: usize) -> Image<f32> {
    Image::from_fn(w, h, |x, y| {
        let fx = frequency(x, w) / w as f32;
        let fy = frequency(y, h) / h as f32;
        (-2.0 * PI * PI * sigma * sigma * (fx * fx + fy * fy)).exp()
    })
}

/// C2b. What the 5-point Laplacian of the filtering unit does to the wave of frequency
/// `(fx, fy)` cycles per pixel: it multiplies it by `L = 4 − 2 cos 2π fx − 2 cos 2π fy`
/// (up to the sign). 0 at frequency 0: a flat image has no roughness.
pub fn laplacian_symbol(fx: f32, fy: f32) -> f32 {
    4.0 - 2.0 * (2.0 * PI * fx).cos() - 2.0 * (2.0 * PI * fy).cos()
}

/// C3. Filter an image in frequency space: transform it, multiply every frequency by the
/// real `gain` at the same pixel (DFT layout), transform back, keep the real part.
pub fn apply_gain(g: &Image<f32>, gain: &Image<f32>) -> Image<f32> {
    let spectrum = fft2_real(g);
    let filtered = spectrum.zip_map(gain, |z, w| z * w);
    fft2(&filtered, true).map(|z| z.re)
}

/// The restoration methods of the lecture, each with its one number.
#[derive(Clone, Copy, Debug)]
pub enum Method {
    Inverse,
    Truncated(f32),
    Wiener(f32),
    TikhonovEnergy(f32),
    TikhonovRoughness(f32),
}

/// C4. Restore the image `g`, blurred with the OTF `otf` (DFT layout), with one of the
/// methods: build the gain at every frequency from C1 (and C2b for the roughness penalty),
/// then filter with C3. The inverse divides by `H` (and gives 0 where `H` is 0).
pub fn restore(g: &Image<f32>, otf: &Image<f32>, method: Method) -> Image<f32> {
    let (w, h) = (g.width(), g.height());
    let gain = otf.map_xy(|x, y, hk| {
        let fx = frequency(x, w) / w as f32;
        let fy = frequency(y, h) / h as f32;
        match method {
            Method::Inverse => truncated_gain(hk, 0.0),
            Method::Truncated(eps) => truncated_gain(hk, eps),
            Method::Wiener(snr) => wiener_gain(hk, snr),
            Method::TikhonovEnergy(lambda) => tikhonov_gain(hk, lambda, 1.0),
            Method::TikhonovRoughness(lambda) => tikhonov_gain(hk, lambda, laplacian_symbol(fx, fy).powi(2)),
        }
    });
    apply_gain(g, &gain)
}
