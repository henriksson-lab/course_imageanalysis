// Model solution for the coding exercises of "Image formation and the point spread function".

use imgcourse::{Border, Image};
use std::f32::consts::PI;

/// C1. The image of a row of point sources: every point puts a copy of the PSF, scaled by
/// its brightness, around its own position, and the image is the sum of the copies.
/// The PSF has odd length; its middle element sits on the point. Light that falls outside
/// the row is lost.
pub fn image_of_points(points: &[f32], psf: &[f32]) -> Vec<f32> {
    let r = (psf.len() / 2) as isize;
    let mut image = vec![0.0; points.len()];
    for (p, &brightness) in points.iter().enumerate() {
        // one scaled copy of the PSF, centred on the point p ("sum of shifted PSFs")
        for (j, &h) in psf.iter().enumerate() {
            let x = p as isize + j as isize - r;
            if x >= 0 && (x as usize) < image.len() {
                image[x as usize] += brightness * h;
            }
        }
    }
    image
}

/// C2a. A Gaussian PSF with standard deviation `sigma` pixels, on a square of
/// `2 · ceil(3 sigma) + 1` pixels with the peak in the middle, normalised to sum 1.
pub fn gaussian_psf(sigma: f32) -> Image<f32> {
    let r = (3.0 * sigma).ceil() as usize;
    let size = 2 * r + 1;
    let g = Image::from_fn(size, size, |x, y| {
        let (dx, dy) = (x as f32 - r as f32, y as f32 - r as f32);
        (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp()
    });
    let total = g.sum();
    g.map(|v| v / total) // sum 1: blur moves light, it does not remove it
}

/// C2b. The PSF of motion along x over `length` pixels (odd): a line, `length` wide and
/// 1 high, every pixel `1 / length`.
pub fn motion_psf(length: usize) -> Image<f32> {
    Image::new(length, 1, 1.0 / length as f32)
}

/// C3. Blur an image with a PSF (any odd width and height, centre in the middle):
/// `out(x, y) = sum over (i, j) of psf(i, j) · img(x − (i − rx), y − (j − ry))`,
/// a convolution. Pixels outside the image are read with the `border` rule.
pub fn blur(img: &Image<f32>, psf: &Image<f32>, border: Border) -> Image<f32> {
    let (rx, ry) = ((psf.width() / 2) as isize, (psf.height() / 2) as isize);
    let mut out = Image::new(img.width(), img.height(), 0.0);
    for (x, y) in img.coords() {
        let mut sum = 0.0;
        for (i, j) in psf.coords() {
            // the light at (x, y) that came from the object point (x − dx, y − dy)
            let (dx, dy) = (i as isize - rx, j as isize - ry);
            sum += psf[(i, j)] * img.get_border(x as isize - dx, y as isize - dy, border);
        }
        out[(x, y)] = sum;
    }
    out
}

/// C4. The OTF of a symmetric 1D PSF (odd length, centre in the middle) at the frequency
/// `k` in cycles per pixel: `H(k) = sum over j of h[j] cos(2π k (j − r))`, `r` the centre.
/// Real, because the PSF is symmetric; it can be negative.
pub fn otf(psf: &[f32], k: f32) -> f32 {
    let r = (psf.len() / 2) as f32;
    psf.iter().enumerate().map(|(j, &h)| h * (2.0 * PI * k * (j as f32 - r)).cos()).sum()
}

/// C5. The MTF of an ideal lens (circular pupil) at the frequency `k` in cycles per µm, for
/// numerical aperture `na` and wavelength `wavelength` in µm. With `s = k / cut-off` and the
/// cut-off `2 NA / λ`: `H = (2/π)(arccos s − s sqrt(1 − s²))` for `s ≤ 1`, else 0.
pub fn lens_mtf(k: f32, na: f32, wavelength: f32) -> f32 {
    let cutoff = 2.0 * na / wavelength;
    let s = (k / cutoff).abs();
    if s >= 1.0 {
        return 0.0; // beyond the cut-off nothing passes
    }
    2.0 / PI * (s.acos() - s * (1.0 - s * s).sqrt())
}
