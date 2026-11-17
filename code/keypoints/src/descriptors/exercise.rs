// Coding exercises for "Describing a keypoint". Replace each `todo!()` with your code,
// then check it with `cargo test -p keypoints descriptors::` and look at `cargo run -p keypoints -- descriptors`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};

pub use crate::common::bilinear;

/// A pixel position `(x, y)` in a patch.
pub type Pixel = (usize, usize);

/// C1. The histogram of one cell: 8 bins of 45°, centred at 0°, 45°, ..., 315°. Every vote
/// `(magnitude, direction in degrees)` goes to the nearest bin (no sharing between bins);
/// directions near 360° belong to the bin at 0°.
///
/// Hint: the nearest bin is `(direction.rem_euclid(360.0) / 45.0).round() as usize % 8`.
pub fn cell_histogram(votes: &[(f32, f32)]) -> [f32; 8] {
    todo!("C1: cell_histogram")
}

/// C2. Normalise to length 1, clip every entry at `clip`, normalise to length 1 again
/// (slide "Gradient histograms, step by step"). A vector of zeros stays zeros.
///
/// Hint: write a small closure `unit` that divides a vector by its length (if not 0),
/// and use it twice.
pub fn normalise_clip(v: &[f32], clip: f32) -> Vec<f32> {
    todo!("C2: normalise_clip")
}

/// C3a. Sample an `n × n` patch in the keypoint frame: centred at `(x, y)`, `12σ` wide,
/// turned by `theta` (degrees, counter-clockwise on the screen). Sample `(i, j)` lies at
/// frame coordinates `u = (i + ½ − n/2) · 12σ/n` along the frame's x axis and `v` likewise
/// along its y axis; read with `bilinear`.
///
/// Hint: with `(s, c) = theta.to_radians().sin_cos()`, the sample lies on the screen at
/// `(x + u c + v s, y − u s + v c)`.
pub fn sample_patch(img: &Image<f32>, x: f32, y: f32, sigma: f32, theta: f32, n: usize) -> Image<f32> {
    todo!("C3a: sample_patch")
}

/// C3b. The normalised patch: all pixels as one vector, minus their mean, divided by its length.
///
/// Hint: `patch.mean()` and `patch.pixels()`.
pub fn normalised_patch(patch: &Image<f32>) -> Vec<f32> {
    todo!("C3b: normalised_patch")
}

/// C4. The gradient-histogram descriptor of a 16 × 16 patch (already sampled in the frame):
/// gradients by central differences, weighted by a Gaussian of σ = 8 px around the patch
/// centre; 4 × 4 cells of 4 × 4 pixels; per cell an 8-bin histogram (C1); concatenated cell by
/// cell, row by row (cell `(cx, cy)` fills entries `8 (4 cy + cx) .. 8 (4 cy + cx) + 8`);
/// then `normalise_clip` with 0.2. 128 numbers.
///
/// Hint: loop over the 16 cells; for each, collect the 16 votes
/// `(magnitude × weight, direction)` and call `cell_histogram`. The direction counter-clockwise
/// on the screen is `(-gy).atan2(gx).to_degrees()`; `d[a..b].copy_from_slice(&hist)` copies a cell in.
pub fn gradient_descriptor(patch: &Image<f32>) -> Vec<f32> {
    todo!("C4: gradient_descriptor")
}

/// C5a. The binary descriptor of a (smoothed) patch: bit `i` is 1 if the patch is darker at
/// `pairs[i].0` than at `pairs[i].1` (both `(x, y)`). The bits are packed into 64-bit words:
/// bit `i` is bit `i % 64` of word `i / 64`.
///
/// Hint: `vec![0u64; pairs.len().div_ceil(64)]`; set bit i with `words[i / 64] |= 1 << (i % 64)`.
pub fn binary_descriptor(patch: &Image<f32>, pairs: &[(Pixel, Pixel)]) -> Vec<u64> {
    todo!("C5a: binary_descriptor")
}

/// C5b. The Hamming distance: the number of bits in which two binary descriptors differ.
///
/// Hint: `(x ^ y).count_ones()` counts the differing bits of two words.
pub fn hamming(a: &[u64], b: &[u64]) -> u32 {
    todo!("C5b: hamming")
}

/// Not an exercise: the Euclidean (L2) distance between two descriptors.
pub fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum::<f32>().sqrt()
}
