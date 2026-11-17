// Coding exercises for "Brightness constancy and the aperture problem". Replace each `todo!()` with your code,
// then check it with `cargo test -p opticalflow flow_constraint::` and look at `cargo run -p opticalflow -- flow-constraint`.
#![allow(unused_variables, unused_imports)]

/// C1. The derivatives of a 1D profile between two frames: `ix[x]` is the central difference
/// `(f1[x+1] − f1[x−1]) / 2` of frame 1 (0 at the two ends), `it[x] = f2[x] − f1[x]`.
///
/// Hint: `vec![0.0; n]` for Ix, then fill `1..n - 1`.
pub fn derivatives_1d(f1: &[f32], f2: &[f32]) -> (Vec<f32>, Vec<f32>) {
    todo!("C1: derivatives_1d")
}

/// C2. The speed seen by one pixel, `u = −It / Ix`, or `None` where the slope is (almost) 0
/// (`|Ix| < 1e-6`).
pub fn pixel_speed(ix: f32, it: f32) -> Option<f32> {
    todo!("C2: pixel_speed")
}

/// C3. The least-squares speed of a window: `u = −Σ Ix It / Σ Ix²` over the pixels
/// `centre − half ..= centre + half` (those that lie inside the profile). `None` if
/// `Σ Ix² < 1e-6` (a flat window).
///
/// Hint: `centre.saturating_sub(half)` stays at 0 instead of going below; the last pixel is
/// `(centre + half).min(ix.len() - 1)`.
pub fn window_speed(ix: &[f32], it: &[f32], centre: usize, half: usize) -> Option<f32> {
    todo!("C3: window_speed")
}

/// C4. The normal flow of one pixel: the point of its constraint line `Ix u + Iy v + It = 0`
/// closest to the origin, `−It ∇I / |∇I|²`. `None` if `|∇I|² < 1e-6`.
pub fn normal_flow(ix: f32, iy: f32, it: f32) -> Option<(f32, f32)> {
    todo!("C4: normal_flow")
}

/// C5. Where the constraint lines of two pixels cross: solve
/// `Ix1 u + Iy1 v = −It1`, `Ix2 u + Iy2 v = −It2` (each pixel given as `[Ix, Iy, It]`).
/// `None` if the gradients are (almost) parallel, `|det| < 1e-6`: the aperture problem.
///
/// Hint: Cramer's rule: `u = (−It1 Iy2 + It2 Iy1) / det`, `v = (−Ix1 It2 + It1 Ix2) / det`.
pub fn intersect(a: [f32; 3], b: [f32; 3]) -> Option<(f32, f32)> {
    todo!("C5: intersect")
}
