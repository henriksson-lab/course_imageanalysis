// Coding exercises for "Noise levels, denoisers and the score". Replace each `todo!()` with your code,
// then check it with `cargo test -p diffusion score_matching::` and look at `cargo run -p diffusion -- score-matching`.
#![allow(unused_variables, unused_imports)]

use imgcourse::fft::{fft2, frequency};
use imgcourse::{Complex, Image};

/// C1. The signal-to-noise ratio and the noise level of a signal level ᾱ:
/// `SNR = ᾱ / (1 − ᾱ)` and `σ = 1 / √SNR`. Returns `(snr, sigma)`.
pub fn noise_level(alpha_bar: f32) -> (f32, f32) {
    todo!("C1: noise_level")
}

/// C2. From a predicted noise `eps` at `x = √ᾱ x0 + √(1 − ᾱ) ε` to the other targets: the
/// clean estimate `x̂0 = (x − √(1 − ᾱ) ε̂) / √ᾱ`, the score `−ε̂ / √(1 − ᾱ)` and the velocity
/// `v = √ᾱ ε̂ − √(1 − ᾱ) x̂0`. Returns `(x0, score, v)`.
pub fn from_noise(x: f32, eps: f32, alpha_bar: f32) -> (f32, f32, f32) {
    todo!("C2: from_noise")
}

/// C3. The binned average of training pairs `(noisy, clean)`: split `lo ... hi` into `bins`
/// equal bins by the noisy value, and average the clean values in each bin. Returns, per bin,
/// `(centre of the bin, mean clean value, number of pairs)`; the mean is 0 for an empty bin.
/// Pairs outside `lo ... hi` are ignored.
pub fn binned_average(pairs: &[(f32, f32)], lo: f32, hi: f32, bins: usize) -> Vec<(f32, f32, usize)> {
    todo!("C3: binned_average")
}

/// C4. The radially averaged power spectrum of an `n × n` image: for every frequency
/// `(fx, fy)` of its DFT, the power `|F|² / n²` goes to the ring `r = round(√(fx² + fy²))`;
/// the result is the mean power of each ring, from `r = 0` to the corners of the spectrum,
/// `r = round(n / √2)`. White noise of variance σ² has power σ² in every ring.
///
/// Hint: `fft2(&img.map(Complex::from), false)` is the DFT; `frequency(k, n)` turns an index
/// into a frequency (negative above n/2).
pub fn radial_power(img: &Image<f32>) -> Vec<f32> {
    todo!("C4: radial_power")
}

/// C5. The Wiener denoiser with a radially symmetric image model: multiply every frequency
/// of the noisy image by the gain `P(r) / (P(r) + σ²)`, with `P` the model's power of its
/// ring (`power[r]`, as from `radial_power`; rings beyond the list get gain 0), and
/// transform back.
pub fn wiener_denoise(noisy: &Image<f32>, sigma: f32, power: &[f32]) -> Image<f32> {
    todo!("C5: wiener_denoise")
}
