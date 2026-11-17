//! Bead images for the chromatic shift (the scene, Pearson's r and the pictures are in
//! `src/common.rs`). Not part of the exercise: read it if you like.

use crate::common::{render, Scene};
use imgcourse::{Image, Rng};

/// A bead image: `n` bright beads that emit in both channels, B shifted by `shift` pixels
/// (the chromatic shift), on the same offset and with the same noise as `p`.
pub fn beads(p: &Scene, n: usize, shift: (f32, f32)) -> (Image<f32>, Image<f32>) {
    let mut rng = Rng::new(p.seed + 1000);
    let s = p.size as f32;
    let pos: Vec<(f32, f32, f32, f32)> = (0..n).map(|_| (8.0 + (s - 16.0) * rng.uniform(), 8.0 + (s - 16.0) * rng.uniform(), 1.0, 0.8)).collect();
    let moved: Vec<_> = pos.iter().map(|&(x, y, r, v)| (x + shift.0, y + shift.1, r, v)).collect();
    let (a, b) = (render(p.size, &pos, p.blur), render(p.size, &moved, p.blur));
    let a = Image::from_fn(p.size, p.size, |x, y| a[(x, y)] + p.offset + p.noise * rng.normal());
    let b = Image::from_fn(p.size, p.size, |x, y| b[(x, y)] + p.offset + p.noise * rng.normal());
    (a, b)
}

