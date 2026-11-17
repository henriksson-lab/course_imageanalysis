// Model solution for the coding exercises of "Richardson–Lucy, 3D, and the PSF you use".

use imgcourse::{Border, Image};

/// C1. Blur a periodic image with a PSF (odd width and height, centre in the middle):
/// `out(x, y) = sum over (i, j) of psf(i, j) · img(x − (i − rx), y − (j − ry))`, reading
/// across the edges with the wrap rule (the image repeats).
pub fn blur(img: &Image<f32>, psf: &Image<f32>) -> Image<f32> {
    let (rx, ry) = ((psf.width() / 2) as isize, (psf.height() / 2) as isize);
    let mut out = Image::new(img.width(), img.height(), 0.0);
    for (x, y) in img.coords() {
        let mut sum = 0.0;
        for (i, j) in psf.coords() {
            let (dx, dy) = (i as isize - rx, j as isize - ry);
            sum += psf[(i, j)] * img.get_border(x as isize - dx, y as isize - dy, Border::Wrap);
        }
        out[(x, y)] = sum;
    }
    out
}

/// The PSF turned by 180° (mirrored in x and y). Given (not an exercise).
pub fn mirror(psf: &Image<f32>) -> Image<f32> {
    let (w, h) = (psf.width(), psf.height());
    Image::from_fn(w, h, |x, y| psf[(w - 1 - x, h - 1 - y)])
}

/// C2. One Richardson–Lucy step, with a constant `background` (counts per pixel):
/// blur the estimate and add the background; divide the data by it, pixel by pixel; spread the
/// ratio back with the mirrored PSF; multiply the estimate by the result.
pub fn rl_step(x: &Image<f32>, y: &Image<f32>, psf: &Image<f32>, background: f32) -> Image<f32> {
    // 1. what the microscope would show if x were true
    let blurred = blur(x, psf).map(|v| v + background);
    // 2. compare with the data as a ratio
    let ratio = y.zip_map(&blurred, |d, b| if b > 0.0 { d / b } else { 0.0 });
    // 3. spread the ratio back: each object pixel collects from the image pixels it lights
    let back = blur(&ratio, &mirror(psf));
    // 4. multiply
    x.zip_map(&back, |a, b| a * b)
}

/// C3. Richardson–Lucy: start from a flat image, every pixel the mean of `y − background`
/// (at least a small positive number), and do `iterations` steps.
pub fn richardson_lucy(y: &Image<f32>, psf: &Image<f32>, iterations: usize, background: f32) -> Image<f32> {
    let start = (y.mean() - background).max(1e-6);
    let mut x = Image::new(y.width(), y.height(), start);
    for _ in 0..iterations {
        x = rl_step(&x, y, psf, background);
    }
    x
}

/// C4. The Poisson deviance per pixel between the counts `y` and the model `m`:
/// the mean over the pixels of `2 (y ln(y / m) − (y − m))`, where `y ln(y / m)` counts as 0
/// for `y = 0`. About 1 when the model explains the data to the noise level.
pub fn poisson_deviance(y: &Image<f32>, m: &Image<f32>) -> f32 {
    let mut sum = 0.0;
    for (&yi, &mi) in y.pixels().iter().zip(m.pixels()) {
        let log_term = if yi > 0.0 { yi * (yi / mi).ln() } else { 0.0 };
        sum += 2.0 * (log_term - (yi - mi));
    }
    sum / y.pixels().len() as f32
}
