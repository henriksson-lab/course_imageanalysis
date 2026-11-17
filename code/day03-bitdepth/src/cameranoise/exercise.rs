// Coding exercises for "Camera noise, SNR and PSNR". Replace each `todo!()` with your
// code, then check it with `cargo test -p day03 cameranoise::` and look at `cargo run -p day03 -- cameranoise`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Image, Rng};

/// A camera: full well (electrons), read noise (electrons, standard deviation) and the number
/// of bits of its ADC, whose range is matched to the full well.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub full_well: f32,
    pub read_noise: f32,
    pub bits: u32,
}

/// C1a. The gain: electrons per ADU, the full well spread over 2^b − 1 steps.
///
/// Hint: `(1u32 << cam.bits) - 1` is 2^b − 1.
pub fn gain(cam: &Camera) -> f32 {
    todo!("C1a: gain")
}

/// C1b. The dynamic range as a ratio: full well over read noise.
pub fn dynamic_range(cam: &Camera) -> f32 {
    todo!("C1b: dynamic_range")
}

/// C1c. An amplitude ratio (signal over noise SD, or a dynamic range) in decibels: 20 log10.
pub fn amplitude_db(ratio: f32) -> f32 {
    todo!("C1c: amplitude_db")
}

/// C2. A random Poisson count with mean `mean` (shot noise). Below a mean of 30, multiply
/// uniform random numbers until the product drops below e^(−mean), and count the steps
/// (Knuth); from 30 on, the Gaussian of the same mean and variance, rounded (never below 0).
///
/// Hint: `rng.uniform()` is uniform in [0, 1), `rng.normal()` Gaussian with mean 0 and SD 1.
/// The count is the number of extra factors needed.
pub fn poisson(mean: f32, rng: &mut Rng) -> u32 {
    todo!("C2: poisson")
}

/// C3. Take a picture: `electrons` is the mean number of electrons each pixel collects. Per pixel:
/// a Poisson count, at most the full well; plus Gaussian read noise; divided by the gain and
/// rounded to whole ADU, kept in 0 … 2^b − 1.
///
/// Hint: `Image::from_fn(w, h, |x, y| ...)` (not `map`: the closure changes `rng`, and
/// `from_fn` allows that).
pub fn expose(cam: &Camera, electrons: &Image<f32>, rng: &mut Rng) -> Image<u32> {
    todo!("C3: expose")
}

/// C4a. The mean and the standard deviation (divide by n) of some values.
pub fn mean_and_sd(values: &[f32]) -> (f32, f32) {
    todo!("C4a: mean_and_sd")
}

/// C4b. The noise the model predicts at a mean signal of `signal` electrons: shot, read and
/// quantisation noise add in variance, √(S + σ_r² + g²/12).
pub fn predicted_noise(cam: &Camera, signal: f32) -> f32 {
    todo!("C4b: predicted_noise")
}

/// C5. The PSNR of an image against a reference, in dB: 10 log10(peak² / MSE).
///
/// Hint: the MSE is the mean of (img − reference)².
pub fn psnr(img: &Image<f32>, reference: &Image<f32>, peak: f32) -> f32 {
    todo!("C5: psnr")
}
