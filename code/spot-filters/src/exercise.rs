// Coding exercises for "Finding spots: filter, then local maxima".
// Replace each `todo!()` with your code, then check it with `cargo test -p spot-filters`
// and look at `cargo run -p spot-filters`.
#![allow(unused_variables, unused_imports)]

use crate::sim::mad_sigma;
use imgcourse::{Border, Image};

/// C1. The local maxima of `img` in windows of radius `r` (a (2r+1) × (2r+1) square): the
/// pixels that are at least as large as every pixel in their window. Pixels outside the image
/// are ignored. Tie rule: of equal values, only the first in raster order (row by row) counts,
/// so a pixel must be **larger** than the pixels before it and **at least as large** as the
/// pixels after it. Returns the positions `(x, y)` in raster order.
///
/// Hint: loop over the window offsets `dx, dy` in `-r..=r` and skip `(0, 0)` and
/// positions outside (`img.contains`). A neighbour comes earlier in raster order if `dy < 0`, or
/// `dy == 0` and `dx < 0`.
pub fn local_maxima(img: &Image<f32>, r: usize) -> Vec<(usize, usize)> {
    todo!("C1: local_maxima")
}

/// C2. The spot template as a filter kernel: a Gaussian of width `sigma` on a square of radius
/// `r = ceil(3 sigma)` (size 2r+1), scaled so that its weights sum to 1. With `zero_mean`, its
/// mean is then subtracted from every weight, so that the weights sum to 0 (centre–surround).
/// The centre of the kernel is pixel `(r, r)`.
///
/// Hint: `Image::from_fn(size, size, |x, y| ...)` with `dx = x - r`, `dy = y - r`; then
/// divide by `k.sum()`, and for `zero_mean` subtract `k.mean()` (both with `map`).
pub fn spot_kernel(sigma: f32, zero_mean: bool) -> Image<f32> {
    todo!("C2: spot_kernel")
}

/// C3. Filter `img` with a kernel of odd size, centred on its middle pixel, without flipping
/// it (correlation, as in the kernels unit). Outside the image: mirror.
///
/// Hint: for every pixel, loop over `kernel.coords()`; the offset of kernel pixel
/// `(kx, ky)` is `(kx - width/2, ky - height/2)`. Read the image with
/// `img.get_border(.., .., Border::Mirror)`.
pub fn correlate(img: &Image<f32>, kernel: &Image<f32>) -> Image<f32> {
    todo!("C3: correlate")
}

/// C4. The à trous decomposition with `levels` levels: `a0 = img`, `a_j` = `a_(j-1)` blurred
/// along rows and then columns with the taps (1, 4, 6, 4, 1)/16 spaced `2^(j-1)` pixels apart
/// (mirror outside), `w_j = a_(j-1) - a_j`. Returns `([w1, w2, ...], a_levels)`.
///
/// Hint: write one pass along x or y with the five taps at offsets
/// `(i - 2) * step`, i = 0..5, and apply it twice (rows, then columns). Keep `a` in a variable
/// and replace it by the blurred image at the end of each level.
pub fn atrous(img: &Image<f32>, levels: usize) -> (Vec<Image<f32>>, Image<f32>) {
    todo!("C4: atrous")
}

/// C5. The multiscale product of the levels `use_levels` (counted from 1, as w1, w2, ...): each
/// level is kept only where it exceeds `k` times its own noise (`mad_sigma` of its values) and
/// set to 0 elsewhere; the kept levels are multiplied pixel by pixel.
///
/// Hint: start from an image of ones; for each chosen level compute
/// `t = k * mad_sigma(w.pixels())` and multiply with `zip_map`.
pub fn multiscale_product(ws: &[Image<f32>], use_levels: &[usize], k: f32) -> Image<f32> {
    todo!("C5: multiscale_product")
}
