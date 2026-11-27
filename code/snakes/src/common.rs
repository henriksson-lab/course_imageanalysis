//! Helpers shared by the units of day 31 (filled in, not exercises): Gaussian blur, bilinear
//! lookup between pixels, and a circle of beads.

use imgcourse::{filters, Border, Image};

/// Gaussian blur of width `sigma` in two passes (unit "Filtering & convolution"), cut at 3σ,
/// copying the edge outside the image. `sigma = 0` returns a copy.
pub fn gaussian_blur(img: &Image<f32>, sigma: f32) -> Image<f32> {
    if sigma <= 0.0 {
        return img.clone();
    }
    let r = (3.0 * sigma).ceil() as isize;
    let mut w: Vec<f32> = (-r..=r).map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let total: f32 = w.iter().sum();
    w.iter_mut().for_each(|v| *v /= total);
    let pass = |src: &Image<f32>, along_x: bool| {
        src.map_xy(|x, y, _| {
            let mut sum = 0.0;
            for d in -r..=r {
                let (xx, yy) = if along_x { (x as isize + d, y as isize) } else { (x as isize, y as isize + d) };
                sum += w[(d + r) as usize] * src.get_border(xx, yy, Border::Clamp);
            }
            sum
        })
    };
    pass(&pass(img, true), false)
}

/// The image value at a position between pixels, by bilinear interpolation (unit "Images as
/// arrays"); outside the image the nearest edge pixel is used.
pub fn bilinear(img: &Image<f32>, x: f32, y: f32) -> f32 {
    filters::bilinear(img, x, y, Border::Clamp)
}

/// `n` points `[x, y]` on a circle of radius `r` around `(cx, cy)`.
pub fn circle(cx: f32, cy: f32, r: f32, n: usize) -> Vec<[f32; 2]> {
    (0..n)
        .map(|i| {
            let t = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
            [cx + r * t.cos(), cy + r * t.sin()]
        })
        .collect()
}
