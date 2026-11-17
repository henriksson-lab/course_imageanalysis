//! Given: the Gaussian blur, written in the edges unit (canny/). Not an exercise here.

use imgcourse::{Border, Image};

/// A 1D Gaussian kernel of width `sigma`, cut at 4σ and scaled to sum 1.
pub fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let r = (4.0 * sigma).ceil() as isize;
    let w: Vec<f32> = (-r..=r).map(|x| (-((x * x) as f32) / (2.0 * sigma * sigma)).exp()).collect();
    let sum: f32 = w.iter().sum();
    w.iter().map(|v| v / sum).collect()
}

/// Filter with the 1D kernel `w` along the rows (`along_x`) or the columns; outside: copy the edge.
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

/// The image blurred with a Gaussian of width `sigma`: rows, then columns.
pub fn gaussian_blur(img: &Image<f32>, sigma: f32) -> Image<f32> {
    let g = gaussian_kernel(sigma);
    filter_1d(&filter_1d(img, &g, true), &g, false)
}
