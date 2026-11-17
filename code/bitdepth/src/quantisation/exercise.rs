// Coding exercises for "Quantisation and bit depth". Replace each `todo!()` with your
// code, then check it with `cargo test -p bitdepth quantisation::` and look at `cargo run -p bitdepth -- quantisation`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Image, Rng};

/// C1a. The code k of the level nearest to `v` (a value in [0, 1]) with `bits` bits:
/// the levels are 0, Δ, 2Δ, …, 1 with Δ = 1/(2^b − 1), so k = round(v / Δ), kept in 0 … 2^b − 1.
///
/// Hint: `(1u32 << bits) - 1` is 2^b − 1; `.round()`, then `.clamp(0.0, top)` before `as u32`.
pub fn quantise(v: f32, bits: u32) -> u32 {
    todo!("C1a: quantise")
}

/// C1b. The stored value of code `k`: k·Δ.
pub fn level(k: u32, bits: u32) -> f32 {
    todo!("C1b: level")
}

/// C1c. Round every pixel to the nearest of the 2^b levels (the stored values, not the codes).
///
/// Hint: `img.map(|v| ...)` with the two functions above.
pub fn quantise_image(img: &Image<f32>, bits: u32) -> Image<f32> {
    todo!("C1c: quantise_image")
}

/// C2a. The root mean square difference between two images of the same size.
pub fn rms_error(a: &Image<f32>, b: &Image<f32>) -> f32 {
    todo!("C2a: rms_error")
}

/// C2b. The signal-to-noise ratio in decibels as in the lecture's math slide: the variance of
/// the clean image over the mean squared error, 10 log10(σ²_signal / MSE).
///
/// Hint: the variance is the mean of (v − mean)²; `x.log10()`.
pub fn snr_db(clean: &Image<f32>, noisy: &Image<f32>) -> f32 {
    todo!("C2b: snr_db")
}

/// C3. Random dither: add noise, uniform between −Δ/2 and +Δ/2, to every pixel, then round
/// to the nearest level. `seed` fixes the noise.
///
/// Hint: `let mut rng = Rng::new(seed);` then `rng.uniform()` is between 0 and 1. Δ is
/// `level(1, bits)`.
pub fn dither_random(img: &Image<f32>, bits: u32, seed: u64) -> Image<f32> {
    todo!("C3: dither_random")
}

/// C4. Error diffusion (Floyd–Steinberg): visit the pixels row by row; round each one, and
/// pass its rounding error (wanted − stored) on to the neighbours not yet visited:
/// 7/16 to the right, 3/16 down-left, 5/16 down, 1/16 down-right (none outside the image).
///
/// Hint: keep a copy of the image (`img.clone()`) to which the errors are added; read the
/// value to round from that copy.
pub fn error_diffusion(img: &Image<f32>, bits: u32) -> Image<f32> {
    todo!("C4: error_diffusion")
}
