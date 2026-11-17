//! Smoothing and sampling tools for the days after they have been taught (the Gaussian kernel and
//! separable filtering on the edges day, bilinear interpolation on day 1).

use crate::{Border, Image};

/// The 1D Gaussian kernel with standard deviation `sigma`, cut at 4σ on each side and scaled
/// so that the weights add up to 1.
pub fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let r = (4.0 * sigma).ceil() as isize;
    let w: Vec<f32> = (-r..=r).map(|x| (-((x * x) as f32) / (2.0 * sigma * sigma)).exp()).collect();
    let sum: f32 = w.iter().sum();
    w.iter().map(|v| v / sum).collect()
}

/// Filter every row (`along_x`) or every column with the odd-length kernel `w`, centred on
/// the pixel; outside the image the nearest edge pixel is copied.
pub fn filter_1d(img: &Image<f32>, w: &[f32], along_x: bool) -> Image<f32> {
    let r = (w.len() / 2) as isize;
    img.map_xy(|x, y, _| {
        let mut sum = 0.0;
        for d in -r..=r {
            let (xx, yy) = if along_x { (x as isize + d, y as isize) } else { (x as isize, y as isize + d) };
            sum += w[(d + r) as usize] * img.get_border(xx, yy, Border::Clamp);
        }
        sum
    })
}

/// Gaussian blur with standard deviation `sigma` (in pixels): rows, then columns.
pub fn gaussian_blur(img: &Image<f32>, sigma: f32) -> Image<f32> {
    let g = gaussian_kernel(sigma);
    filter_1d(&filter_1d(img, &g, true), &g, false)
}

/// The image at a position between pixels, by bilinear interpolation of the 4 surrounding
/// pixels; outside the image the `border` rule decides what is read.
pub fn bilinear(img: &Image<f32>, x: f32, y: f32, border: Border) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (a, b) = (x - x0, y - y0);
    let (x0, y0) = (x0 as isize, y0 as isize);
    let i = |dx: isize, dy: isize| img.get_border(x0 + dx, y0 + dy, border);
    (1.0 - a) * (1.0 - b) * i(0, 0) + a * (1.0 - b) * i(1, 0) + (1.0 - a) * b * i(0, 1) + a * b * i(1, 1)
}

/// Root-mean-square difference of two images of the same size.
pub fn rmse(a: &Image<f32>, b: &Image<f32>) -> f32 {
    let sum: f64 = a.pixels().iter().zip(b.pixels()).map(|(&p, &q)| ((p - q) as f64).powi(2)).sum();
    (sum / a.pixels().len() as f64).sqrt() as f32
}

/// Peak signal-to-noise ratio in dB of `result` against `truth`: 10 log10(peak² / mean squared error).
pub fn psnr(result: &Image<f32>, truth: &Image<f32>, peak: f32) -> f32 {
    let e = rmse(result, truth);
    20.0 * (peak / e).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_and_blur() {
        let g = gaussian_kernel(1.0);
        assert_eq!(g.len(), 9);
        assert!((g.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let flat = Image::new(10, 7, 0.3f32);
        assert!(gaussian_blur(&flat, 2.0).pixels().iter().all(|v| (v - 0.3).abs() < 1e-6));
    }

    #[test]
    fn bilinear_and_errors() {
        let img = Image::from_rows(&[[10.0f32, 20.0], [30.0, 60.0]]);
        assert!((bilinear(&img, 0.25, 0.5, Border::Clamp) - 25.0).abs() < 1e-5);
        let a = Image::new(4, 4, 0.5f32);
        let b = Image::new(4, 4, 0.6f32);
        assert!((rmse(&a, &b) - 0.1).abs() < 1e-6);
        assert!((psnr(&a, &b, 1.0) - 20.0).abs() < 1e-3);
    }
}
