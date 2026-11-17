// Coding exercises for "All smoothers on one image". Replace each `todo!()` with your
// code, then check it with `cargo test -p smoothers` and look at `cargo run -p smoothers`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};

/// C1. The peak signal-to-noise ratio of `result` against the true image, in dB, for grey
/// values in [0, 1]: `10 log10(1 / mean squared error)`.
pub fn psnr(result: &Image<f32>, truth: &Image<f32>) -> f32 {
    todo!("C1: psnr")
}

/// C2. The median filter: every output pixel is the median of the `(2 radius + 1)²` window
/// around it (window pixels outside the image are left out; for an even count, the mean of
/// the two middle values).
///
/// Hint: collect the window values in a `Vec<f32>`, sort it with
/// `window.sort_by(|a, b| a.total_cmp(b))`, and take the middle.
pub fn median_filter(img: &Image<f32>, radius: usize) -> Image<f32> {
    todo!("C2: median_filter")
}

/// C3a. Soft thresholding (B6): move `c` towards 0 by `t`, and to 0 if it is closer than `t`.
pub fn soft_threshold(c: f32, t: f32) -> f32 {
    todo!("C3a: soft_threshold")
}

/// C3b. Wavelet shrinkage with one level of the 2D Haar transform: for every 2 × 2 block
/// `a b / c d` compute the mean part `(a+b+c+d)/2` and the three details
/// `(a-b+c-d)/2` (across), `(a+b-c-d)/2` (down) and `(a-b-c+d)/2` (diagonal); soft-threshold
/// the three details with `t`, and transform back. The width and height must be even.
///
/// Hint: loop over the blocks with `(0..height).step_by(2)` and `(0..width).step_by(2)`. Going back,
/// `a = (s + h + v + g) / 2`, `b = (s - h + v - g) / 2`, `c = (s + h - v - g) / 2`, `d = (s - h - v + g) / 2`.
pub fn haar_shrink(img: &Image<f32>, t: f32) -> Image<f32> {
    todo!("C3b: haar_shrink")
}

/// C4. How many pixels a threshold gets wrong: the number of pixels where `result > level`
/// and `truth > level` disagree (slide "Smoothing before segmentation").
pub fn threshold_errors(result: &Image<f32>, truth: &Image<f32>, level: f32) -> usize {
    todo!("C4: threshold_errors")
}

/// Not an exercise: a Gaussian blur (two passes of a sampled, normalised 1D Gaussian of
/// radius 4σ, mirror border).
pub fn gaussian_blur(img: &Image<f32>, sigma: f32) -> Image<f32> {
    let r = (4.0 * sigma).ceil() as isize;
    let w: Vec<f32> = (-r..=r).map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let total: f32 = w.iter().sum();
    let pass = |src: &Image<f32>, along_x: bool| {
        src.map_xy(|x, y, _| {
            let mut sum = 0.0;
            for d in -r..=r {
                let (xx, yy) = if along_x { (x as isize + d, y as isize) } else { (x as isize, y as isize + d) };
                sum += w[(d + r) as usize] * src.get_border(xx, yy, Border::Mirror);
            }
            sum / total
        })
    };
    pass(&pass(img, true), false)
}
