// Model solution for the coding exercises of "Describing a keypoint".

use imgcourse::{Border, Image};

pub use crate::common::bilinear;

/// A pixel position `(x, y)` in a patch.
pub type Pixel = (usize, usize);

/// C1. The histogram of one cell: 8 bins of 45°, centred at 0°, 45°, ..., 315°. Every vote
/// `(magnitude, direction in degrees)` goes to the nearest bin (no sharing between bins);
/// directions near 360° belong to the bin at 0°.
pub fn cell_histogram(votes: &[(f32, f32)]) -> [f32; 8] {
    let mut hist = [0.0; 8];
    for &(magnitude, direction) in votes {
        let bin = (direction.rem_euclid(360.0) / 45.0).round() as usize % 8;
        hist[bin] += magnitude;
    }
    hist
}

/// C2. Normalise to length 1, clip every entry at `clip`, normalise to length 1 again
/// (slide "Gradient histograms, step by step"). A vector of zeros stays zeros.
pub fn normalise_clip(v: &[f32], clip: f32) -> Vec<f32> {
    let unit = |v: &[f32]| {
        let length = v.iter().map(|a| a * a).sum::<f32>().sqrt();
        v.iter().map(|a| if length > 0.0 { a / length } else { 0.0 }).collect::<Vec<f32>>()
    };
    let clipped: Vec<f32> = unit(v).iter().map(|a| a.min(clip)).collect();
    unit(&clipped)
}

/// C3a. Sample an `n × n` patch in the keypoint frame: centred at `(x, y)`, `12σ` wide,
/// turned by `theta` (degrees, counter-clockwise on the screen). Sample `(i, j)` lies at
/// frame coordinates `u = (i + ½ − n/2) · 12σ/n` along the frame's x axis and `v` likewise
/// along its y axis; read with `bilinear`.
pub fn sample_patch(img: &Image<f32>, x: f32, y: f32, sigma: f32, theta: f32, n: usize) -> Image<f32> {
    let spacing = 12.0 * sigma / n as f32;
    let (s, c) = theta.to_radians().sin_cos();
    Image::from_fn(n, n, |i, j| {
        let u = (i as f32 + 0.5 - n as f32 / 2.0) * spacing;
        let v = (j as f32 + 0.5 - n as f32 / 2.0) * spacing;
        // The frame's x axis points along (cos θ, −sin θ) on screen (y down), its y axis along (sin θ, cos θ).
        bilinear(img, x + u * c + v * s, y - u * s + v * c)
    })
}

/// C3b. The normalised patch: all pixels as one vector, minus their mean, divided by its length.
pub fn normalised_patch(patch: &Image<f32>) -> Vec<f32> {
    let mean = patch.mean();
    let centred: Vec<f32> = patch.pixels().iter().map(|v| v - mean).collect();
    let length = centred.iter().map(|a| a * a).sum::<f32>().sqrt();
    centred.iter().map(|a| if length > 0.0 { a / length } else { 0.0 }).collect()
}

/// C4. The gradient-histogram descriptor of a 16 × 16 patch (already sampled in the frame):
/// gradients by central differences, weighted by a Gaussian of σ = 8 px around the patch
/// centre; 4 × 4 cells of 4 × 4 pixels; per cell an 8-bin histogram (C1); concatenated cell by
/// cell, row by row (cell `(cx, cy)` fills entries `8 (4 cy + cx) .. 8 (4 cy + cx) + 8`);
/// then `normalise_clip` with 0.2. 128 numbers.
pub fn gradient_descriptor(patch: &Image<f32>) -> Vec<f32> {
    let mut d = vec![0.0; 128];
    for cy in 0..4 {
        for cx in 0..4 {
            let mut votes = Vec::new();
            for y in 4 * cy..4 * cy + 4 {
                for x in 4 * cx..4 * cx + 4 {
                    let p = |dx: isize, dy: isize| patch.get_border(x as isize + dx, y as isize + dy, Border::Clamp);
                    let gx = (p(1, 0) - p(-1, 0)) / 2.0;
                    let gy = (p(0, 1) - p(0, -1)) / 2.0;
                    let (rx, ry) = (x as f32 + 0.5 - 8.0, y as f32 + 0.5 - 8.0);
                    let weight = (-(rx * rx + ry * ry) / (2.0 * 8.0 * 8.0)).exp();
                    // Directions counter-clockwise on screen (y points down).
                    votes.push(((gx * gx + gy * gy).sqrt() * weight, (-gy).atan2(gx).to_degrees()));
                }
            }
            let hist = cell_histogram(&votes);
            d[8 * (4 * cy + cx)..8 * (4 * cy + cx) + 8].copy_from_slice(&hist);
        }
    }
    normalise_clip(&d, 0.2)
}

/// C5a. The binary descriptor of a (smoothed) patch: bit `i` is 1 if the patch is darker at
/// `pairs[i].0` than at `pairs[i].1` (both `(x, y)`). The bits are packed into 64-bit words:
/// bit `i` is bit `i % 64` of word `i / 64`.
pub fn binary_descriptor(patch: &Image<f32>, pairs: &[(Pixel, Pixel)]) -> Vec<u64> {
    let mut words = vec![0u64; pairs.len().div_ceil(64)];
    for (i, &(p, q)) in pairs.iter().enumerate() {
        if patch[p] < patch[q] {
            words[i / 64] |= 1 << (i % 64);
        }
    }
    words
}

/// C5b. The Hamming distance: the number of bits in which two binary descriptors differ.
pub fn hamming(a: &[u64], b: &[u64]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

/// Not an exercise: the Euclidean (L2) distance between two descriptors.
pub fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum::<f32>().sqrt()
}
