// Model solution for the coding exercises of "Noise levels, denoisers and the score".

use imgcourse::fft::{fft2, frequency};
use imgcourse::{Complex, Image};

/// C1. The signal-to-noise ratio and the noise level of a signal level ᾱ:
/// `SNR = ᾱ / (1 − ᾱ)` and `σ = 1 / √SNR`. Returns `(snr, sigma)`.
pub fn noise_level(alpha_bar: f32) -> (f32, f32) {
    let snr = alpha_bar / (1.0 - alpha_bar);
    (snr, 1.0 / snr.sqrt())
}

/// C2. From a predicted noise `eps` at `x = √ᾱ x0 + √(1 − ᾱ) ε` to the other targets: the
/// clean estimate `x̂0 = (x − √(1 − ᾱ) ε̂) / √ᾱ`, the score `−ε̂ / √(1 − ᾱ)` and the velocity
/// `v = √ᾱ ε̂ − √(1 − ᾱ) x̂0`. Returns `(x0, score, v)`.
pub fn from_noise(x: f32, eps: f32, alpha_bar: f32) -> (f32, f32, f32) {
    let (a, b) = (alpha_bar.sqrt(), (1.0 - alpha_bar).sqrt());
    let x0 = (x - b * eps) / a;
    (x0, -eps / b, a * eps - b * x0)
}

/// C3. The binned average of training pairs `(noisy, clean)`: split `lo ... hi` into `bins`
/// equal bins by the noisy value, and average the clean values in each bin. Returns, per bin,
/// `(centre of the bin, mean clean value, number of pairs)`; the mean is 0 for an empty bin.
/// Pairs outside `lo ... hi` are ignored.
pub fn binned_average(pairs: &[(f32, f32)], lo: f32, hi: f32, bins: usize) -> Vec<(f32, f32, usize)> {
    let width = (hi - lo) / bins as f32;
    let mut sum = vec![0.0f32; bins];
    let mut count = vec![0usize; bins];
    for &(x, x0) in pairs {
        if x < lo || x >= hi {
            continue;
        }
        let b = (((x - lo) / width) as usize).min(bins - 1);
        sum[b] += x0;
        count[b] += 1;
    }
    (0..bins).map(|b| (lo + (b as f32 + 0.5) * width, if count[b] > 0 { sum[b] / count[b] as f32 } else { 0.0 }, count[b])).collect()
}

/// C4. The radially averaged power spectrum of an `n × n` image: for every frequency
/// `(fx, fy)` of its DFT, the power `|F|² / n²` goes to the ring `r = round(√(fx² + fy²))`;
/// the result is the mean power of each ring, from `r = 0` to the corners of the spectrum,
/// `r = round(n / √2)`. White noise of variance σ² has power σ² in every ring.
pub fn radial_power(img: &Image<f32>) -> Vec<f32> {
    let n = img.width();
    let spectrum = fft2(&img.map(Complex::from), false);
    let rings = (n as f32 / 2f32.sqrt()).round() as usize + 1;
    let mut sum = vec![0.0f32; rings];
    let mut count = vec![0usize; rings];
    for (kx, ky) in spectrum.coords() {
        let (fx, fy) = (frequency(kx, n), frequency(ky, n));
        let r = (fx * fx + fy * fy).sqrt().round() as usize;
        sum[r] += spectrum[(kx, ky)].norm_sqr() / (n * n) as f32;
        count[r] += 1;
    }
    (0..rings).map(|r| sum[r] / count[r].max(1) as f32).collect()
}

/// C5. The Wiener denoiser with a radially symmetric image model: multiply every frequency
/// of the noisy image by the gain `P(r) / (P(r) + σ²)`, with `P` the model's power of its
/// ring (`power[r]`, as from `radial_power`; rings beyond the list get gain 0), and
/// transform back.
pub fn wiener_denoise(noisy: &Image<f32>, sigma: f32, power: &[f32]) -> Image<f32> {
    let n = noisy.width();
    let mut spectrum = fft2(&noisy.map(Complex::from), false);
    for (kx, ky) in noisy.coords() {
        let (fx, fy) = (frequency(kx, n), frequency(ky, n));
        let r = (fx * fx + fy * fy).sqrt().round() as usize;
        let gain = if r < power.len() { power[r] / (power[r] + sigma * sigma) } else { 0.0 };
        spectrum[(kx, ky)] = spectrum[(kx, ky)] * gain;
    }
    fft2(&spectrum, true).map(|c| c.re)
}
