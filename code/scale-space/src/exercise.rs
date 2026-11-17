// Coding exercises for "Scale space, blobs and pyramids". Replace each `todo!()` with your
// code, then check it with `cargo test -p scale-space` and look at `cargo run -p scale-space`.
#![allow(unused_variables, unused_imports)]

use crate::gaussian_blur;
use imgcourse::{Border, Image};

/// C1a. The scale-normalised Laplacian of Gaussian, sign flipped so that bright blobs are
/// positive: −σ² (Lxx + Lyy), where L is the image blurred at `sigma` and the second
/// derivatives are the differences [1 −2 1] along x and along y (copy the edge outside).
///
/// Hint: `gaussian_blur(img, sigma)` (in given.rs) gives L. Then, at every pixel, read the
/// neighbours with `l.get_border(.., .., Border::Clamp)`.
pub fn normalised_log(img: &Image<f32>, sigma: f32) -> Image<f32> {
    todo!("C1a: normalised_log")
}

/// C1b. The zero crossings of a row of values (Marr–Hildreth): every place where the sign
/// changes between element `i` and `i + 1`, as a position between them by linear
/// interpolation, `i + a / (a − b)` for the values `a` and `b`.
///
/// Hint: a sign change means one value > 0 and the other < 0 (a value of exactly 0 is not
/// a crossing here).
pub fn zero_crossings(row: &[f32]) -> Vec<f32> {
    todo!("C1b: zero_crossings")
}

/// One detected blob: its centre, its scale, and the normalised response there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Blob {
    pub x: usize,
    pub y: usize,
    pub sigma: f32,
    pub response: f32,
}

/// C2. Blob detection: `stack[k]` is the normalised LoG at scale `sigmas[k]` (all the same
/// size). A blob is a pixel whose response is above `threshold` and larger than all its
/// neighbours in (x, y, σ): 8 in its own scale, 9 in the scale below and 9 above (only 9 + 8
/// at the smallest and the largest scale). Pixels on the edge of the image are skipped: they
/// do not have all their neighbours, and the copied edge makes their second differences wrong.
///
/// Hint: loop over k and over the inner pixels, x in 1 .. width − 1 and y in 1 .. height − 1.
/// Then three loops over dk, dy, dx in −1 ..= 1; skip the pixel itself and scales outside the
/// stack. A neighbour that is ≥ the pixel rules it out.
pub fn detect_blobs(stack: &[Image<f32>], sigmas: &[f32], threshold: f32) -> Vec<Blob> {
    todo!("C2: detect_blobs")
}

/// C3. The difference of Gaussians: the image blurred at `k·sigma` minus the image blurred
/// at `sigma`. It is close to (k − 1) σ² ∇²L, i.e. to −(k − 1) times `normalised_log`.
pub fn dog(img: &Image<f32>, sigma: f32, k: f32) -> Image<f32> {
    todo!("C3: dog")
}

/// C4a. One level down the Gaussian pyramid: blur with σ = 1, then keep every second pixel
/// in each direction (pixels (0, 0), (2, 0), …). A 2w × 2h image becomes w × h.
///
/// Hint: `img.width().div_ceil(2)` is the new width.
pub fn downsample(img: &Image<f32>) -> Image<f32> {
    todo!("C4a: downsample")
}

/// C4b. Back up to a `width × height` image: pixel (x, y) reads the small image at
/// (x/2, y/2) by bilinear interpolation (copy the edge outside).
///
/// Hint: as bilinear interpolation in the images unit: the corner `(sx.floor(), sy.floor())`,
/// the fractions `a = sx − sx.floor()` and `b`, and the weights (1−a)(1−b), a(1−b), (1−a)b, ab.
pub fn upsample(small: &Image<f32>, width: usize, height: usize) -> Image<f32> {
    todo!("C4b: upsample")
}

/// C4c. The Laplacian pyramid with `levels` levels: level i is g_i − up(g_(i+1)), where g_0 is
/// the image and g_(i+1) = downsample(g_i); the last level is the small Gaussian level itself.
pub fn laplacian_pyramid(img: &Image<f32>, levels: usize) -> Vec<Image<f32>> {
    todo!("C4c: laplacian_pyramid")
}

/// C4d. Rebuild the image from its Laplacian pyramid: start with the last level, and going
/// back up, upsample and add the level: g_i = l_i + up(g_(i+1)).
///
/// Hint: walk the levels from the second last to the first: `pyramid[..n - 1].iter().rev()`.
pub fn reconstruct(pyramid: &[Image<f32>]) -> Image<f32> {
    todo!("C4d: reconstruct")
}

