// Undersamples the k-space of a synthetic head phantom in three ways, reconstructs by zero
// filling and by compressed sensing, and writes the images to out/cs-mri/.
//   cargo run -p day11 -- cs-mri                      (your code)
//   cargo run -p day11 --features solution -- cs-mri  (the model solution)

use crate::cs_mri::*;
use imgcourse::fft::{fft2_real, frequency};
use imgcourse::*;

pub fn main() {
    let dir = "out/cs-mri";
    let n = 256;
    let truth = phantom(n, false);
    save_gray(&truth, format!("{dir}/phantom.png"));

    // Three masks with about a quarter of the lines, and one with an eighth.
    let centre_only: Vec<bool> = (0..n).map(|k| frequency(k, n).abs() < 32.0).collect();
    let every_2nd: Vec<bool> = (0..n).map(|k| k % 2 == 0 || frequency(k, n).abs() <= 8.0).collect();
    let random4 = variable_density_mask(n, n / 4, n / 16, 3.0, 1);
    let random8 = variable_density_mask(n, n / 8, n / 16, 3.0, 1);
    let lambda = 0.01;
    let levels = 5;
    let iterations = 100;
    for (name, mask) in [("centre", &centre_only), ("every2nd", &every_2nd), ("random4x", &random4), ("random8x", &random8)] {
        let lines = mask.iter().filter(|m| **m).count();
        // The mask as a picture, with frequency 0 in the middle (row n/2), as k-space is usually shown.
        save_gray(&Image::from_fn(n, n, |_, y| if mask[(y + n / 2) % n] { 1.0 } else { 0.0 }), format!("{dir}/mask-{name}.png"));
        let zf = zero_filled(&truth, mask);
        save_gray(&zf, format!("{dir}/zero-filled-{name}.png"));
        let cs = cs_reconstruct(&measure(&truth, mask), mask, lambda, levels, iterations);
        save_gray(&cs, format!("{dir}/cs-{name}.png"));
        println!(
            "{name:9}: {lines:3} of {n} lines ({:.1}×), zero filled {:.1} dB, CS {:.1} dB",
            n as f32 / lines as f32,
            psnr(&zf, &truth),
            psnr(&cs, &truth)
        );
    }

    // Five small faint lesions: shown with a narrow grey window (0.15 to 0.40).
    let lesions = phantom(n, true);
    let window = |img: &Image<f32>| img.map(|v| ((v - 0.15) / 0.25).clamp(0.0, 1.0));
    save_gray(&window(&lesions), format!("{dir}/lesions.png"));
    let cs = cs_reconstruct(&measure(&lesions, &random4), &random4, lambda, levels, iterations);
    save_gray(&window(&cs), format!("{dir}/lesions-cs-random4x.png"));
    save_gray(&window(&zero_filled(&lesions, &random4)), format!("{dir}/lesions-zero-filled-random4x.png"));
    println!("lesion phantom, random4x: CS {:.1} dB", psnr(&cs, &lesions));
    println!("wrote the results to {dir}/");
}

/// The measured k-space: the 2D DFT of `img` on the lines of the mask, 0 elsewhere.
fn measure(img: &Image<f32>, mask: &[bool]) -> Image<Complex> {
    let mut k = fft2_real(img);
    for (kx, ky) in k.coords() {
        if !mask[ky] {
            k[(kx, ky)] = Complex::ZERO;
        }
    }
    k
}

/// Peak signal-to-noise ratio in dB (peak 1).
fn psnr(a: &Image<f32>, truth: &Image<f32>) -> f32 {
    let mse = a.zip_map(truth, |p, q| (p - q) * (p - q)).mean();
    10.0 * (1.0 / mse).log10()
}

/// The modified Shepp–Logan head phantom, `n × n`, values 0 to 1. With `lesions`, five small
/// faint discs (3 pixels across at n = 256, contrast 0.02 to 0.05) are added.
fn phantom(n: usize, lesions: bool) -> Image<f32> {
    // (value, half axis a, half axis b, centre x, centre y, angle in degrees); y points up.
    let mut ellipses = vec![
        (1.0, 0.69, 0.92, 0.0, 0.0, 0.0),
        (-0.8, 0.6624, 0.874, 0.0, -0.0184, 0.0),
        (-0.2, 0.11, 0.31, 0.22, 0.0, -18.0),
        (-0.2, 0.16, 0.41, -0.22, 0.0, 18.0),
        (0.1, 0.21, 0.25, 0.0, 0.35, 0.0),
        (0.1, 0.046, 0.046, 0.0, 0.1, 0.0),
        (0.1, 0.046, 0.046, 0.0, -0.1, 0.0),
        (0.1, 0.046, 0.023, -0.08, -0.605, 0.0),
        (0.1, 0.023, 0.023, 0.0, -0.606, 0.0),
        (0.1, 0.023, 0.046, 0.06, -0.605, 0.0),
    ];
    if lesions {
        let r = 1.5 / (n as f32 / 2.0);
        for (c, x, y) in [(0.05, -0.45, -0.35), (0.04, 0.45, -0.35), (0.03, -0.5, 0.45), (0.025, 0.5, 0.45), (0.02, 0.0, -0.35)] {
            ellipses.push((c, r, r, x, y, 0.0));
        }
    }
    Image::from_fn(n, n, |px, py| {
        let u = 2.0 * (px as f32 + 0.5) / n as f32 - 1.0;
        let v = 1.0 - 2.0 * (py as f32 + 0.5) / n as f32;
        let mut value = 0.0;
        for &(c, a, b, x0, y0, deg) in &ellipses {
            let t = (deg as f32).to_radians();
            let (du, dv) = (u - x0, v - y0);
            let (p, q) = (du * t.cos() + dv * t.sin(), -du * t.sin() + dv * t.cos());
            if (p / a) * (p / a) + (q / b) * (q / b) <= 1.0 {
                value += c;
            }
        }
        value
    })
}
