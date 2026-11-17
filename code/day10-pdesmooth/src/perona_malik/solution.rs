// Model solution for the coding exercises of "Edge-stopping diffusion (Perona–Malik)".

use imgcourse::{Border, Image};

/// C1a. The edge-stopping function `g1(s) = exp(-(s/K)^2)`.
pub fn g1(s: f32, k: f32) -> f32 {
    (-(s / k).powi(2)).exp()
}

/// C1b. The edge-stopping function `g2(s) = 1 / (1 + (s/K)^2)`.
pub fn g2(s: f32, k: f32) -> f32 {
    1.0 / (1.0 + (s / k).powi(2))
}

/// An edge-stopping function: `g(size of the difference, K)`, for example `g1` or `g2`.
pub type Stop = fn(f32, f32) -> f32;

/// C2. One Perona–Malik step on a row (slide "One step by hand"): between every pair of
/// neighbours the flow is `g(|d|) * d`, `d` = the difference; every value gets `dt` times
/// the flows coming in. Mirror border: nothing flows out at the two ends.
pub fn pm_step_1d(row: &[f32], dt: f32, k: f32, g: Stop) -> Vec<f32> {
    let n = row.len();
    // flow[i]: from pair (i, i+1), positive when i+1 is the larger one (heat comes to i)
    let mut flow = vec![0.0; n.saturating_sub(1)];
    for i in 0..n.saturating_sub(1) {
        let d = row[i + 1] - row[i];
        flow[i] = g(d.abs(), k) * d;
    }
    let mut out = row.to_vec();
    for i in 0..n {
        if i + 1 < n {
            out[i] += dt * flow[i]; // from the right neighbour
        }
        if i > 0 {
            out[i] -= dt * flow[i - 1]; // to the left neighbour
        }
    }
    out
}

/// C3. One Perona–Malik step on an image: the four flows of the heat step, each multiplied
/// by `g(|difference|)`. With `sigma > 0` the conductivity is computed from the differences
/// of the image blurred with a Gaussian of that σ (slide "Price 2: ill-posed, and the cure"),
/// while the flows still use the differences of the image itself.
pub fn pm_step(img: &Image<f32>, dt: f32, k: f32, g: Stop, sigma: f32) -> Image<f32> {
    let smooth = if sigma > 0.0 { gaussian_blur(img, sigma) } else { img.clone() };
    let b = Border::Mirror;
    let mut out = Image::new(img.width(), img.height(), 0.0);
    for (x, y) in img.coords() {
        let (xi, yi) = (x as isize, y as isize);
        let mut change = 0.0;
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let d = img.get_border(xi + dx, yi + dy, b) - img[(x, y)];
            let ds = smooth.get_border(xi + dx, yi + dy, b) - smooth[(x, y)];
            change += g(ds.abs(), k) * d;
        }
        out[(x, y)] = img[(x, y)] + dt * change;
    }
    out
}

/// C4. The peak signal-to-noise ratio of `result` against the true image `truth`, in dB,
/// for grey values with a peak of `peak` (1 for the course's images, 255 for 8-bit numbers):
/// `10 log10(peak^2 / mean squared error)`.
pub fn psnr(result: &Image<f32>, truth: &Image<f32>, peak: f32) -> f32 {
    let mut sum = 0.0;
    for (x, y) in truth.coords() {
        let e = result[(x, y)] - truth[(x, y)];
        sum += e * e;
    }
    let mse = sum / (truth.width() * truth.height()) as f32;
    10.0 * (peak * peak / mse).log10()
}

/// Not an exercise: a Gaussian blur (radius 4σ, mirror border), in `src/common.rs`.
pub use crate::common::gaussian_blur;
