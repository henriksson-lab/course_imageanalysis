// Coding exercises for "Image formation and the point spread function". Replace each `todo!()`
// with your code, then check it with `cargo test -p imageformation psf::` and look at `cargo run -p imageformation -- psf`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};
use std::f32::consts::PI;

/// C1. The image of a row of point sources: every point puts a copy of the PSF, scaled by
/// its brightness, around its own position, and the image is the sum of the copies.
/// The PSF has odd length; its middle element sits on the point. Light that falls outside
/// the row is lost.
///
/// Hint: start from a row of zeros; for every point `p` and every PSF element `j`, add
/// `brightness · psf[j]` to position `p + j − r`, where `r = psf.len() / 2`, if that
/// position is inside the row.
pub fn image_of_points(points: &[f32], psf: &[f32]) -> Vec<f32> {
    todo!("C1: image_of_points")
}

/// C2a. A Gaussian PSF with standard deviation `sigma` pixels, on a square of
/// `2 · ceil(3 sigma) + 1` pixels with the peak in the middle, normalised to sum 1.
///
/// Hint: `Image::from_fn` with `exp(−(dx² + dy²) / 2σ²)`, where `(dx, dy)` is the offset from
/// the middle pixel; then divide every pixel by the sum (`map`).
pub fn gaussian_psf(sigma: f32) -> Image<f32> {
    todo!("C2a: gaussian_psf")
}

/// C2b. The PSF of motion along x over `length` pixels (odd): a line, `length` wide and
/// 1 high, every pixel `1 / length`.
pub fn motion_psf(length: usize) -> Image<f32> {
    todo!("C2b: motion_psf")
}

/// C3. Blur an image with a PSF (any odd width and height, centre in the middle):
/// `out(x, y) = sum over (i, j) of psf(i, j) · img(x − (i − rx), y − (j − ry))`,
/// a convolution. Pixels outside the image are read with the `border` rule.
///
/// Hint: `rx = psf.width() / 2`, `ry = psf.height() / 2`. Note the minus signs: the PSF is
/// flipped, unlike the filters of the filtering unit, so that a point comes out as the PSF.
pub fn blur(img: &Image<f32>, psf: &Image<f32>, border: Border) -> Image<f32> {
    todo!("C3: blur")
}

/// C4. The OTF of a symmetric 1D PSF (odd length, centre in the middle) at the frequency
/// `k` in cycles per pixel: `H(k) = sum over j of h[j] cos(2π k (j − r))`, `r` the centre.
/// Real, because the PSF is symmetric; it can be negative.
pub fn otf(psf: &[f32], k: f32) -> f32 {
    todo!("C4: otf")
}

/// C5. The MTF of an ideal lens (circular pupil) at the frequency `k` in cycles per µm, for
/// numerical aperture `na` and wavelength `wavelength` in µm. With `s = k / cut-off` and the
/// cut-off `2 NA / λ`: `H = (2/π)(arccos s − s sqrt(1 − s²))` for `s ≤ 1`, else 0.
pub fn lens_mtf(k: f32, na: f32, wavelength: f32) -> f32 {
    todo!("C5: lens_mtf")
}
