// Model solution for the coding exercises of "Camera noise, SNR and PSNR".

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
pub fn gain(cam: &Camera) -> f32 {
    cam.full_well / ((1u32 << cam.bits) - 1) as f32
}

/// C1b. The dynamic range as a ratio: full well over read noise.
pub fn dynamic_range(cam: &Camera) -> f32 {
    cam.full_well / cam.read_noise
}

/// C1c. An amplitude ratio (signal over noise SD, or a dynamic range) in decibels: 20 log10.
pub fn amplitude_db(ratio: f32) -> f32 {
    20.0 * ratio.log10()
}

/// C2. A random Poisson count with mean `mean` (shot noise). Below a mean of 30, multiply
/// uniform random numbers until the product drops below e^(−mean), and count the steps
/// (Knuth); from 30 on, the Gaussian of the same mean and variance, rounded (never below 0).
pub fn poisson(mean: f32, rng: &mut Rng) -> u32 {
    if mean < 30.0 {
        let limit = (-mean).exp();
        let mut product = rng.uniform();
        let mut k = 0;
        while product > limit {
            product *= rng.uniform();
            k += 1;
        }
        k
    } else {
        (mean + mean.sqrt() * rng.normal()).round().max(0.0) as u32
    }
}

/// C3. Take a picture: `electrons` is the mean number of electrons each pixel collects. Per pixel:
/// a Poisson count, at most the full well; plus Gaussian read noise; divided by the gain and
/// rounded to whole ADU, kept in 0 … 2^b − 1.
pub fn expose(cam: &Camera, electrons: &Image<f32>, rng: &mut Rng) -> Image<u32> {
    let g = gain(cam);
    let top = ((1u32 << cam.bits) - 1) as f32;
    Image::from_fn(electrons.width(), electrons.height(), |x, y| {
        let count = (poisson(electrons[(x, y)], rng) as f32).min(cam.full_well);
        let charge = count + cam.read_noise * rng.normal();
        (charge / g).round().clamp(0.0, top) as u32
    })
}

/// C4a. The mean and the standard deviation (divide by n) of some values.
pub fn mean_and_sd(values: &[f32]) -> (f32, f32) {
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    (mean, var.sqrt())
}

/// C4b. The noise the model predicts at a mean signal of `signal` electrons: shot, read and
/// quantisation noise add in variance, √(S + σ_r² + g²/12).
pub fn predicted_noise(cam: &Camera, signal: f32) -> f32 {
    let g = gain(cam);
    (signal + cam.read_noise * cam.read_noise + g * g / 12.0).sqrt()
}

/// C5. The PSNR of an image against a reference, in dB: 10 log10(peak² / MSE).
pub fn psnr(img: &Image<f32>, reference: &Image<f32>, peak: f32) -> f32 {
    let n = img.pixels().len() as f32;
    let mse = img.pixels().iter().zip(reference.pixels()).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / n;
    10.0 * (peak * peak / mse).log10()
}
