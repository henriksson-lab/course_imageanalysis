// Wavelet shrinkage on a synthetic shapes image and on the micrograph: thresholds, hard against
// soft, the noise estimate, and cycle spinning; pictures in out/shrinkage/.
//   cargo run -p shrinkage                      (your code)
//   cargo run -p shrinkage --features solution  (the model solution)

use imgcourse::*;
use shrinkage::*;

/// Shapes on a grey background, grey levels 0..255: a disc, a rectangle, a diamond, an ellipse.
fn shapes() -> Image<f32> {
    Image::from_fn(256, 256, |x, y| {
        let (x, y) = (x as f32 + 0.5, y as f32 + 0.5);
        if (x - 75.0).hypot(y - 75.0) < 45.0 {
            210.0
        } else if (150.0..225.0).contains(&x) && (40.0..110.0).contains(&y) {
            50.0
        } else if (x - 80.0).abs() + (y - 185.0).abs() < 50.0 {
            170.0
        } else if ((x - 185.0) / 50.0).powi(2) + ((y - 185.0) / 28.0).powi(2) < 1.0 {
            240.0
        } else {
            110.0
        }
    })
}

/// PSNR with peak 255.
fn psnr(a: &Image<f32>, b: &Image<f32>) -> f32 {
    let mse: f32 = a.pixels().iter().zip(b.pixels()).map(|(p, q)| (p - q) * (p - q)).sum::<f32>() / a.pixels().len() as f32;
    10.0 * (255.0f32 * 255.0 / mse).log10()
}

/// σ̂ from the finest diagonal band (the bottom-right quarter of one level).
fn sigma_hat(img: &Image<f32>, h: &[f32]) -> f32 {
    let c = dwt2d(img, h, 1);
    let (w, ht) = (img.width() / 2, img.height() / 2);
    median_sigma(c.crop(w, ht, w, ht).pixels())
}

fn main() {
    let dir = "out/shrinkage";
    let truth = shapes();
    let sigma = 20.0;
    let noisy = samples::add_noise(&truth, sigma, 5);
    let levels = 4;
    let n = 256 * 256;
    save_gray(&truth.map(|v| v / 255.0), format!("{dir}/shapes.png"));
    save_gray(&noisy.map(|v| v / 255.0), format!("{dir}/noisy.png"));
    let s = sigma_hat(&noisy, &DB4);
    let universal = universal_threshold(s, n);
    println!("shapes, noise σ = {sigma}: noisy PSNR {:.1} dB, σ̂ = {s:.2}, universal threshold {universal:.1} = {:.2} σ̂", psnr(&noisy, &truth), universal / s);

    // Universal threshold, soft and hard (db4, 4 levels).
    for (name, use_soft) in [("soft", true), ("hard", false)] {
        let out = denoise(&noisy, &DB4, levels, universal, use_soft);
        println!("  universal threshold, {name}: PSNR {:.1} dB", psnr(&out, &truth));
        if use_soft {
            save_gray(&out.map(|v| v / 255.0), format!("{dir}/universal-soft.png"));
        }
    }
    // The best threshold for each, on a grid of t / σ̂.
    for (name, use_soft) in [("soft", true), ("hard", false)] {
        let mut best = (0.0, 0.0);
        let mut line = String::new();
        for k in 2..=24 {
            let factor = 0.25 * k as f32;
            let p = psnr(&denoise(&noisy, &DB4, levels, factor * s, use_soft), &truth);
            if p > best.1 {
                best = (factor, p);
            }
            if k % 4 == 0 {
                line += &format!("  {factor:.0}σ̂: {p:.1}");
            }
        }
        println!("  {name}: best t = {:.2} σ̂, PSNR {:.1} dB   ({})", best.0, best.1, line.trim());
        let out = denoise(&noisy, &DB4, levels, best.0 * s, use_soft);
        save_gray(&out.map(|v| v / 255.0), format!("{dir}/best-{name}.png"));
        save_gray_signed(&noisy.zip_map(&out, |a, b| a - b), format!("{dir}/removed-{name}.png"));
    }

    // Cycle spinning with Haar, soft, t = 3 σ̂.
    let s_haar = sigma_hat(&noisy, &HAAR);
    for shifts in [1, 2, 4] {
        let out = cycle_spin(&noisy, &HAAR, levels, 3.0 * s_haar, true, shifts);
        println!("  Haar, soft, t = 3σ̂, {:2} shifts: PSNR {:.1} dB", shifts * shifts, psnr(&out, &truth));
        if shifts != 2 {
            // The disc, enlarged 3 times.
            let crop = out.map(|v| v / 255.0).crop(20, 20, 110, 110);
            save_gray(&Image::from_fn(330, 330, |x, y| crop[(x / 3, y / 3)]), format!("{dir}/haar-shifts{}.png", shifts * shifts));
        }
    }

    // The micrograph: texture against noise.
    let cell = samples::cell().map(|v| 255.0 * v);
    println!("\nmicrograph:");
    println!("  clean image: σ̂ = {:.1} (there is no noise)", sigma_hat(&cell, &DB4));
    for (sigma, factor) in [(10.0, 1.25), (40.0, 1.25)] {
        let noisy = samples::add_noise(&cell, sigma, 6);
        let s = sigma_hat(&noisy, &DB4);
        let out = denoise(&noisy, &DB4, levels, factor * s, true);
        println!("  noise σ = {sigma}: σ̂ = {s:.1}; soft, t = {factor} σ̂: PSNR {:.1} → {:.1} dB", psnr(&noisy, &cell), psnr(&out, &cell));
        if sigma == 40.0 {
            save_gray(&noisy.map(|v| v / 255.0), format!("{dir}/cell-noisy40.png"));
            save_gray(&out.map(|v| v / 255.0), format!("{dir}/cell-denoised40.png"));
        }
    }
    println!("wrote the pictures to {dir}/");
}
