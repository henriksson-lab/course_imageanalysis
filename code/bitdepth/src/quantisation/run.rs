// Rounds a ramp, smooth blobs and the course micrograph to a few bits, with and without
// dither; results in out/quantisation/.
//   cargo run -p bitdepth -- quantisation                      (your code)
//   cargo run -p bitdepth --features solution -- quantisation  (the model solution)

use imgcourse::*;
use crate::quantisation::*;

/// A smooth image: a few wide Gaussian blobs on a gentle slope, values in [0, 1].
fn blobs() -> Image<f32> {
    let spots = [(60.0, 70.0, 40.0, 0.5), (180.0, 90.0, 55.0, 0.4), (110.0, 190.0, 45.0, 0.45)];
    Image::from_fn(256, 256, |x, y| {
        let (xf, yf) = (x as f32, y as f32);
        let mut v = 0.1 + 0.2 * yf / 256.0;
        for (cx, cy, s, a) in spots {
            v += a * (-((xf - cx).powi(2) + (yf - cy).powi(2)) / (2.0 * s * s)).exp();
        }
        v.min(1.0)
    })
}

/// The error image, 0 as mid-grey and ±Δ as white/black.
fn error_picture(img: &Image<f32>, q: &Image<f32>, bits: u32) -> Image<f32> {
    let delta = level(1, bits);
    q.zip_map(img, |a, b| 0.5 + 0.5 * (a - b) / delta)
}

pub fn main() {
    let dir = "out/quantisation";

    // The ramp: RMS error and SNR for 1 to 8 bits.
    let ramp = Image::from_fn(256, 64, |x, _| x as f32 / 255.0);
    let fine = Image::from_fn(256, 256, |x, y| (y * 256 + x) as f32 / 65535.0); // 65 536 different values
    println!("bits   RMS error   Δ/√12      SNR");
    for bits in 1..=8 {
        let q = quantise_image(&fine, bits);
        let delta = level(1, bits);
        println!("{bits:4}   {:.5}     {:.5}    {:5.1} dB", rms_error(&fine, &q), delta / 12f32.sqrt(), snr_db(&fine, &q));
    }
    save_gray(&quantise_image(&ramp, 4), format!("{dir}/ramp-4bit.png"));
    save_gray(&error_picture(&ramp, &quantise_image(&ramp, 4), 4), format!("{dir}/ramp-4bit-error.png"));

    // Smooth blobs at 4 bits: banding, and what dither does to it.
    let img = blobs();
    save_gray(&img, format!("{dir}/blobs.png"));
    let plain = quantise_image(&img, 4);
    let dithered = dither_random(&img, 4, 1);
    let diffused = error_diffusion(&img, 4);
    for (name, q) in [("plain", &plain), ("dither", &dithered), ("diffusion", &diffused)] {
        save_gray(q, format!("{dir}/blobs-4bit-{name}.png"));
        // The eye averages neighbours: compare after a small blur as well.
        let blur = |i: &Image<f32>| i.map_xy(|x, y, _| {
            let mut s = 0.0;
            for dy in -2..=2isize {
                for dx in -2..=2isize {
                    s += i.get_border(x as isize + dx, y as isize + dy, Border::Clamp);
                }
            }
            s / 25.0
        });
        println!("blobs, 4 bits, {name:9}: RMS error {:.4}; after a 5 × 5 average of both: {:.4}", rms_error(&img, q), rms_error(&blur(&img), &blur(q)));
    }

    // The micrograph at 2 bits (4 grey levels).
    let cell = samples::cell();
    for (name, q) in [("plain", quantise_image(&cell, 2)), ("dither", dither_random(&cell, 2, 2)), ("diffusion", error_diffusion(&cell, 2))] {
        println!("micrograph, 2 bits, {name:9}: RMS error {:.4}", rms_error(&cell, &q));
        save_gray(&q, format!("{dir}/cell-2bit-{name}.png"));
    }
    println!("wrote the results to {dir}/");
}
