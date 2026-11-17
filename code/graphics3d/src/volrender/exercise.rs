// Coding exercises for "Seeing voxels: projections and volume rendering".
// Replace each `todo!()` with your code, then check it with `cargo test -p graphics3d volrender::`
// and look at `cargo run -p graphics3d -- volrender`.
// The tasks are called R1, R2, ... because the exercise page already uses C1, C2, ... for the
// demo questions. Voxel (i, j, k) has its centre at the voxel coordinates (i, j, k); in the
// world (millimetres) it sits at (i Δx, j Δy, k Δz).
#![allow(unused_variables, unused_imports)]

use imgcourse::Volume;

/// A colour, red, green and blue in [0, 1].
pub type Rgb = [f32; 3];

/// R1. Trilinear interpolation at the voxel coordinates `(x, y, z)`: the weighted mean of the
/// 8 voxels around the point, with weights from the distances along each axis (as bilinear
/// interpolation in the images unit, one dimension more). Voxels outside the volume count as 0.
///
/// Hint: with `x0 = x.floor()` and `tx = x - x0` (likewise for y and z), loop over
/// dx, dy, dz in 0..2. The weight along x is `1 − tx` for dx = 0 and `tx` for dx = 1. Skip
/// corners outside the volume (`vol.contains(i, j, k)` takes signed numbers).
pub fn trilinear(vol: &Volume<f32>, x: f32, y: f32, z: f32) -> f32 {
    todo!("R1: trilinear")
}

/// R2. The values along a ray: `n` samples at the world points `start + s · step · dir`,
/// s = 0, 1, ..., n − 1 (`dir` has length 1, all in millimetres). Each point is turned into
/// voxel coordinates by dividing by the voxel `spacing`, then sampled with `trilinear`.
///
/// Hint: the voxel coordinates of a world point p are `p[0] / spacing[0]`, and so on.
pub fn sample_ray(vol: &Volume<f32>, spacing: [f32; 3], start: [f32; 3], dir: [f32; 3], step: f32, n: usize) -> Vec<f32> {
    todo!("R2: sample_ray")
}

/// R3a. Front-to-back compositing (slide "From the integral to compositing"): start with
/// C = 0 and T = 1; for each sample (colour c, opacity α), from the eye outwards,
/// C ← C + T α c and T ← T (1 − α). Stop early once T < `eps` (use `eps = 0` for no early
/// stop). Returns the colour, the transmittance left, and the number of samples used.
///
/// Hint: check `t < eps` at the start of each step, before using the sample.
pub fn front_to_back(samples: &[(Rgb, f32)], eps: f32) -> (Rgb, f32, usize) {
    todo!("R3a: front_to_back")
}

/// R3b. Back-to-front compositing (the painter's order): start with C = 0 at the far end and
/// walk towards the eye, C ← α c + (1 − α) C.
///
/// Hint: `samples.iter().rev()` walks from the far end towards the eye.
pub fn back_to_front(samples: &[(Rgb, f32)]) -> Rgb {
    todo!("R3b: back_to_front")
}

/// R4. Opacity correction: an opacity `alpha0` meant for the reference step, used with a
/// step `ratio` times as long: `1 − (1 − alpha0)^ratio`.
pub fn opacity_correct(alpha0: f32, ratio: f32) -> f32 {
    todo!("R4: opacity_correct")
}

/// R5. The first place along a ray where the values cross the isovalue `c` from below:
/// for the first pair of neighbouring samples with `v[i] < c ≤ v[i + 1]`, the position
/// `i + (c − v[i]) / (v[i + 1] − v[i])` (in samples), by linear interpolation. `None` if the
/// ray never crosses `c`.
///
/// Hint: loop over `i` in `0..values.len() - 1` (careful with an empty slice: `saturating_sub`).
pub fn iso_crossing(values: &[f32], c: f32) -> Option<f32> {
    todo!("R5: iso_crossing")
}
