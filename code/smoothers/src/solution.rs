// Model solution for the coding exercises of "All smoothers on one image".

use imgcourse::{Border, Image};

/// C1. The peak signal-to-noise ratio of `result` against the true image, in dB, for grey
/// values in [0, 1]: `10 log10(1 / mean squared error)`.
pub fn psnr(result: &Image<f32>, truth: &Image<f32>) -> f32 {
    let mut sum = 0.0;
    for (x, y) in truth.coords() {
        let e = result[(x, y)] - truth[(x, y)];
        sum += e * e;
    }
    let mse = sum / (truth.width() * truth.height()) as f32;
    10.0 * (1.0 / mse).log10()
}

/// C2. The median filter: every output pixel is the median of the `(2 radius + 1)²` window
/// around it (window pixels outside the image are left out; for an even count, the mean of
/// the two middle values).
pub fn median_filter(img: &Image<f32>, radius: usize) -> Image<f32> {
    let r = radius as isize;
    let mut out = Image::new(img.width(), img.height(), 0.0);
    let mut window = Vec::new();
    for (x, y) in img.coords() {
        window.clear();
        for dy in -r..=r {
            for dx in -r..=r {
                let (qx, qy) = (x as isize + dx, y as isize + dy);
                if img.contains(qx, qy) {
                    window.push(img[(qx as usize, qy as usize)]);
                }
            }
        }
        window.sort_by(|a, b| a.total_cmp(b));
        let n = window.len();
        out[(x, y)] = if n % 2 == 1 { window[n / 2] } else { 0.5 * (window[n / 2 - 1] + window[n / 2]) };
    }
    out
}

/// C3a. Soft thresholding (B6): move `c` towards 0 by `t`, and to 0 if it is closer than `t`.
pub fn soft_threshold(c: f32, t: f32) -> f32 {
    if c > t {
        c - t
    } else if c < -t {
        c + t
    } else {
        0.0
    }
}

/// C3b. Wavelet shrinkage with one level of the 2D Haar transform: for every 2 × 2 block
/// `a b / c d` compute the mean part `(a+b+c+d)/2` and the three details
/// `(a-b+c-d)/2` (across), `(a+b-c-d)/2` (down) and `(a-b-c+d)/2` (diagonal); soft-threshold
/// the three details with `t`, and transform back. The width and height must be even.
pub fn haar_shrink(img: &Image<f32>, t: f32) -> Image<f32> {
    let mut out = img.clone();
    for by in (0..img.height()).step_by(2) {
        for bx in (0..img.width()).step_by(2) {
            let (a, b) = (img[(bx, by)], img[(bx + 1, by)]);
            let (c, d) = (img[(bx, by + 1)], img[(bx + 1, by + 1)]);
            let s = (a + b + c + d) / 2.0;
            let h = soft_threshold((a - b + c - d) / 2.0, t);
            let v = soft_threshold((a + b - c - d) / 2.0, t);
            let g = soft_threshold((a - b - c + d) / 2.0, t);
            // the transform is its own inverse
            out[(bx, by)] = (s + h + v + g) / 2.0;
            out[(bx + 1, by)] = (s - h + v - g) / 2.0;
            out[(bx, by + 1)] = (s + h - v - g) / 2.0;
            out[(bx + 1, by + 1)] = (s - h - v + g) / 2.0;
        }
    }
    out
}

/// C4. How many pixels a threshold gets wrong: the number of pixels where `result > level`
/// and `truth > level` disagree (slide "Smoothing before segmentation").
pub fn threshold_errors(result: &Image<f32>, truth: &Image<f32>, level: f32) -> usize {
    let mut wrong = 0;
    for (x, y) in truth.coords() {
        if (result[(x, y)] > level) != (truth[(x, y)] > level) {
            wrong += 1;
        }
    }
    wrong
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
