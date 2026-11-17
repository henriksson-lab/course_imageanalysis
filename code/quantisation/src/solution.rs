// Model solution for the coding exercises of "Quantisation and bit depth".

use imgcourse::{Image, Rng};

/// C1a. The code k of the level nearest to `v` (a value in [0, 1]) with `bits` bits:
/// the levels are 0, Δ, 2Δ, …, 1 with Δ = 1/(2^b − 1), so k = round(v / Δ), kept in 0 … 2^b − 1.
pub fn quantise(v: f32, bits: u32) -> u32 {
    let top = ((1u32 << bits) - 1) as f32;
    (v * top).round().clamp(0.0, top) as u32
}

/// C1b. The stored value of code `k`: k·Δ.
pub fn level(k: u32, bits: u32) -> f32 {
    k as f32 / ((1u32 << bits) - 1) as f32
}

/// C1c. Round every pixel to the nearest of the 2^b levels (the stored values, not the codes).
pub fn quantise_image(img: &Image<f32>, bits: u32) -> Image<f32> {
    img.map(|v| level(quantise(v, bits), bits))
}

/// C2a. The root mean square difference between two images of the same size.
pub fn rms_error(a: &Image<f32>, b: &Image<f32>) -> f32 {
    let sum: f32 = a.pixels().iter().zip(b.pixels()).map(|(p, q)| (p - q) * (p - q)).sum();
    (sum / a.pixels().len() as f32).sqrt()
}

/// C2b. The signal-to-noise ratio in decibels as in the lecture's math slide: the variance of
/// the clean image over the mean squared error, 10 log10(σ²_signal / MSE).
pub fn snr_db(clean: &Image<f32>, noisy: &Image<f32>) -> f32 {
    let n = clean.pixels().len() as f32;
    let mean = clean.mean();
    let var: f32 = clean.pixels().iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let mse = rms_error(clean, noisy).powi(2);
    10.0 * (var / mse).log10()
}

/// C3. Random dither: add noise, uniform between −Δ/2 and +Δ/2, to every pixel, then round
/// to the nearest level. `seed` fixes the noise.
pub fn dither_random(img: &Image<f32>, bits: u32, seed: u64) -> Image<f32> {
    let delta = level(1, bits);
    let mut rng = Rng::new(seed);
    Image::from_fn(img.width(), img.height(), |x, y| {
        let noisy = img[(x, y)] + delta * (rng.uniform() - 0.5);
        level(quantise(noisy, bits), bits)
    })
}

/// C4. Error diffusion (Floyd–Steinberg): visit the pixels row by row; round each one, and
/// pass its rounding error (wanted − stored) on to the neighbours not yet visited:
/// 7/16 to the right, 3/16 down-left, 5/16 down, 1/16 down-right (none outside the image).
pub fn error_diffusion(img: &Image<f32>, bits: u32) -> Image<f32> {
    let mut wanted = img.clone(); // the values plus the errors passed on so far
    let mut out = Image::new(img.width(), img.height(), 0.0f32);
    for (x, y) in img.coords() {
        let stored = level(quantise(wanted[(x, y)], bits), bits);
        out[(x, y)] = stored;
        let err = wanted[(x, y)] - stored;
        for (dx, dy, w) in [(1, 0, 7.0), (-1, 1, 3.0), (0, 1, 5.0), (1, 1, 1.0)] {
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            if wanted.contains(nx, ny) {
                wanted[(nx as usize, ny as usize)] += err * w / 16.0;
            }
        }
    }
    out
}
