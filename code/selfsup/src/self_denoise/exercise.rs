// Coding exercises for "Self-supervised denoising". Replace each `todo!()` with your code,
// then check it with `cargo test -p selfsup self_denoise::` and look at `cargo run -p selfsup -- self-denoise`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{linalg, Border, Image};

/// A linear denoising filter: `out(p) = Σ weights[k] · in(p + offset k) + bias` over the
/// `(2 radius + 1)²` window around `p`, row by row. A blind-spot filter has centre weight 0.
#[derive(Clone, Debug)]
pub struct Filter {
    pub radius: usize,
    pub weights: Vec<f32>,
    pub bias: f32,
}

impl Filter {
    /// The weight of the offset `(dx, dy)` from the centre.
    pub fn weight(&self, dx: isize, dy: isize) -> f32 {
        let n = 2 * self.radius as isize + 1;
        self.weights[((dy + self.radius as isize) * n + dx + self.radius as isize) as usize]
    }
}

/// C1. The grey values of the `(2 radius + 1)²` window around `(x, y)`, row by row (mirror
/// border). With `blind = true` the centre pixel itself is left out (one value fewer).
///
/// Hint: two loops over `dy` and `dx` from `-r` to `r`; skip `dx == 0 && dy == 0` when blind;
/// `img.get_border(x as isize + dx, y as isize + dy, Border::Mirror)` reads with the mirror rule.
pub fn neighbourhood(img: &Image<f32>, x: usize, y: usize, radius: usize, blind: bool) -> Vec<f32> {
    todo!("C1: neighbourhood")
}

/// C2. Train a linear filter by least squares: for every pixel, the prediction from the
/// window of `input` around it (without the centre if `blind`) should match `target` at
/// that pixel. One row per pixel: the window values and a 1 (for the bias).
/// Noise2Void: `input` = `target` = the noisy image, `blind = true`.
/// Noise2Noise: `input` = one noisy image, `target` = a second noisy image of the same thing.
///
/// Hint: collect the rows as `Vec<Vec<f64>>` and the targets as `Vec<f64>`, then
/// `linalg::least_squares(&rows, &b)` gives the weights and, last, the bias. For a blind-spot
/// filter, put a 0 into `weights` at the centre (index `n / 2` of the `n` window positions),
/// so that `weights` always has `(2 radius + 1)²` entries.
pub fn fit_filter(input: &Image<f32>, target: &Image<f32>, radius: usize, blind: bool) -> Filter {
    todo!("C2: fit_filter")
}

/// C3. Apply the filter: every output pixel is the weighted sum of the window around it
/// (mirror border) plus the bias.
///
/// Hint: `img.map_xy(|x, y, _| ...)` with `neighbourhood(img, x, y, f.radius, false)`.
pub fn apply_filter(img: &Image<f32>, f: &Filter) -> Image<f32> {
    todo!("C3: apply_filter")
}

/// C4. Put the pixel's own value back. The blind-spot prediction `pred` and the noisy pixel
/// are two independent estimates of the clean value; with the noise level `sigma` known
/// (from a calibration of the camera), mix them: `out = pred + c · (noisy − pred)` with
/// `c = 1 − sigma² / mean((noisy − pred)²)` (not below 0). Returns the image and `c`.
///
/// Hint: first the mean of the squared differences over all pixels, then
/// `pred.zip_map(noisy, |p, y| p + c * (y - p))`.
pub fn mix_back(noisy: &Image<f32>, pred: &Image<f32>, sigma: f32) -> (Image<f32>, f32) {
    todo!("C4: mix_back")
}
