// Coding exercises for "Richardson–Lucy, 3D, and the PSF you use". Replace each `todo!()`
// with your code, then check it with `cargo test -p deconvolution richardson_lucy::` and look at `cargo run -p deconvolution -- richardson-lucy`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};

/// C1. Blur a periodic image with a PSF (odd width and height, centre in the middle):
/// `out(x, y) = sum over (i, j) of psf(i, j) · img(x − (i − rx), y − (j − ry))`, reading
/// across the edges with the wrap rule (the image repeats).
///
/// Hint: as `blur` in the PSF unit, reading with `img.get_border(..., Border::Wrap)`.
pub fn blur(img: &Image<f32>, psf: &Image<f32>) -> Image<f32> {
    todo!("C1: blur")
}

/// The PSF turned by 180° (mirrored in x and y). Given (not an exercise).
pub fn mirror(psf: &Image<f32>) -> Image<f32> {
    let (w, h) = (psf.width(), psf.height());
    Image::from_fn(w, h, |x, y| psf[(w - 1 - x, h - 1 - y)])
}

/// C2. One Richardson–Lucy step, with a constant `background` (counts per pixel):
/// blur the estimate and add the background; divide the data by it, pixel by pixel; spread the
/// ratio back with the mirrored PSF; multiply the estimate by the result.
///
/// Hint: four lines, one per step: `blur(x, psf).map(...)`, `y.zip_map(&blurred, ...)`,
/// `blur(&ratio, &mirror(psf))`, `x.zip_map(&back, ...)`. Where the blurred estimate is 0, use a
/// ratio of 0.
pub fn rl_step(x: &Image<f32>, y: &Image<f32>, psf: &Image<f32>, background: f32) -> Image<f32> {
    todo!("C2: rl_step")
}

/// C3. Richardson–Lucy: start from a flat image, every pixel the mean of `y − background`
/// (at least a small positive number), and do `iterations` steps.
///
/// Hint: `Image::new(w, h, start)`, then `x = rl_step(&x, y, psf, background)` in a loop.
pub fn richardson_lucy(y: &Image<f32>, psf: &Image<f32>, iterations: usize, background: f32) -> Image<f32> {
    todo!("C3: richardson_lucy")
}

/// C4. The Poisson deviance per pixel between the counts `y` and the model `m`:
/// the mean over the pixels of `2 (y ln(y / m) − (y − m))`, where `y ln(y / m)` counts as 0
/// for `y = 0`. About 1 when the model explains the data to the noise level.
pub fn poisson_deviance(y: &Image<f32>, m: &Image<f32>) -> f32 {
    todo!("C4: poisson_deviance")
}
