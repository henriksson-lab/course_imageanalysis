//! Test images that need no files.

use crate::{Image, Rng};

/// The course micrograph: a transmission electron micrograph of a mesenchymal stem cell,
/// 256 × 256, grey values in `[0, 1]`. Image: Robert M. Hunt, CC BY 3.0, via Wikimedia Commons.
pub fn cell() -> Image<f32> {
    crate::io::decode_gray(include_bytes!("../data/cell256.png"))
}

/// A white letter F on black, `size × size`: easy to see if an image was mirrored or turned.
pub fn letter_f(size: usize) -> Image<f32> {
    let s = size as f32;
    Image::from_fn(size, size, |x, y| {
        let (u, v) = (x as f32 / s, y as f32 / s);
        let stem = (0.25..0.40).contains(&u) && (0.15..0.85).contains(&v);
        let top = (0.25..0.75).contains(&u) && (0.15..0.30).contains(&v);
        let middle = (0.25..0.62).contains(&u) && (0.45..0.58).contains(&v);
        if stem || top || middle { 1.0 } else { 0.0 }
    })
}

/// A vertical step edge: `left` on the left half, `right` on the right half.
pub fn step(width: usize, height: usize, left: f32, right: f32) -> Image<f32> {
    Image::from_fn(width, height, |x, _| if x < width / 2 { left } else { right })
}

/// A checkerboard of `square × square` blocks, 0 and 1.
pub fn checker(width: usize, height: usize, square: usize) -> Image<f32> {
    Image::from_fn(width, height, |x, y| ((x / square + y / square) % 2) as f32)
}

/// A copy of `img` with Gaussian noise of standard deviation `sigma` added (not clipped).
pub fn add_noise(img: &Image<f32>, sigma: f32, seed: u64) -> Image<f32> {
    let mut rng = Rng::new(seed);
    Image::from_fn(img.width(), img.height(), |x, y| img[(x, y)] + sigma * rng.normal())
}
