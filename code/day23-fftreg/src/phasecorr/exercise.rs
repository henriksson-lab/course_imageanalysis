// Coding exercises for "Phase correlation". Replace each `todo!()` with your code,
// then check it with `cargo test -p day23 phasecorr::` and look at `cargo run -p day23 -- phasecorr`.
#![allow(unused_variables, unused_imports)]

use imgcourse::fft::{fft, fft2, frequency};
use imgcourse::{Complex, Image};

/// A small number added to the sizes before dividing, so that empty frequencies do not get
/// a full vote (and nothing is divided by zero).
pub const EPS: f32 = 1e-6;

/// C1. Phase correlation of two signals of the same length: transform both, take the
/// cross-power `A_k · conj(B_k)`, divide it by its size (plus `EPS`), transform back and keep
/// the real part (slide "Keep only the phase").
///
/// Hint: as the cross-correlation of the last unit, but divide every product `c` by
/// `c.abs() + EPS` before the inverse transform.
pub fn phase_correlation_1d(a: &[f32], b: &[f32]) -> Vec<f32> {
    todo!("C1: phase_correlation_1d")
}

/// C2. The Hann taper of length `n`: `w(i) = ½ (1 − cos(2π i / (n − 1)))`, 0 at both ends and
/// 1 in the middle.
///
/// Hint: `(0..n).map(|i| ...).collect()`; `std::f32::consts::PI`.
pub fn hann(n: usize) -> Vec<f32> {
    todo!("C2: hann")
}

/// Settings of the 2D phase correlation.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// Divide the cross-power by its size to the power `p`: 0 is cross-correlation, 1 phase
    /// correlation.
    pub p: f32,
    /// Multiply both images by the 2D Hann taper (after subtracting their means).
    pub taper: bool,
    /// If not 0: weight the normalised cross-power by `exp(−f² / (2 sd²))`, where `f` is the
    /// frequency in cycles per pixel (a low-pass weight).
    pub lowpass_sd: f32,
}

/// C3. Phase correlation of two images of the same size (slides "What it costs: noise" and
/// "Tapering"): subtract each image's mean, taper (optional), FFT both, divide `A_k conj(B_k)`
/// by `|A_k conj(B_k)|^p + EPS`, apply the low-pass weight (optional), inverse FFT, real part.
/// `out[(tx, ty)]` is the score of the shift `(tx, ty)`, wrapped around as in the last unit.
///
/// Hint: prepare each image as an `Image<Complex>` with `map_xy` (mean subtracted, times
/// `wx[x] * wy[y]` if tapering), `fft2` both, then `zip_map` for the product and `map_xy` for
/// the division and the weight. `frequency(kx, w) / w as f32` is the frequency in cycles per
/// pixel; `c.abs().powf(s.p)`.
pub fn phase_correlation(a: &Image<f32>, b: &Image<f32>, s: Settings) -> Image<f32> {
    todo!("C3: phase_correlation")
}

/// C4. The highest peak of a correlation surface and how much to trust it: returns
/// `(tx, ty, height, ratio)`, the shift (signed, as in the last unit), the peak's value, and
/// its ratio to the second-highest **local maximum** (a pixel larger than its 8 neighbours,
/// wrapping around at the edges) (slide "How much to trust a peak").
///
/// Hint: first collect all local maxima `(value, x, y)`: compare with the 8 neighbours
/// `((x + dx) % w, (y + dy) % h)` for `dx, dy` in `w - 1, 0, 1` (that is −1, 0, 1 with
/// wrap-around). Sort them, largest first.
pub fn peak_and_ratio(c: &Image<f32>) -> (isize, isize, f32, f32) {
    todo!("C4: peak_and_ratio")
}
