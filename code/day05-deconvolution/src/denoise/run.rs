// Denoises a test image (a step, a ramp crossed by a thin line, a lattice of dots and one unique
// spot) with your median, bilateral and non-local means filters, and a Gaussian blur for
// comparison. Prints the PSNR and writes the results and the method noise to out/denoise/.
//   cargo run -p day05 -- denoise                      (your code)
//   cargo run -p day05 --features solution -- denoise  (the model solution)

use crate::denoise::*;
use imgcourse::*;

/// The test image, 128 × 128, grey levels / 255.
fn test_image() -> Image<f32> {
    Image::from_fn(128, 128, |x, y| {
        let v = match (x < 64, y < 64) {
            (true, true) => if x < 32 { 60.0 } else { 180.0 },                    // a step
            (false, true) => {
                let ramp = 40.0 + 140.0 * (x - 64) as f32 / 63.0;                  // a ramp,
                if x == 96 { ramp + 80.0 } else { ramp }                          // crossed by a 1-pixel line
            }
            (true, false) => {
                let (dx, dy) = ((x % 6) as f32 - 2.5, (y % 6) as f32 - 2.5);   // a lattice of dots
                if dx * dx + dy * dy < 2.6 { 200.0 } else { 60.0 }
            }
            (false, false) => {
                let (dx, dy) = (x as f32 - 96.0, y as f32 - 96.0);              // one unique spot
                if dx * dx + dy * dy < 5.0 { 220.0 } else { 100.0 }
            }
        };
        v / 255.0
    })
}

/// A Gaussian blur, for comparison (the filtering unit).
fn gaussian(img: &Image<f32>, sigma: f32) -> Image<f32> {
    let r = (3.0 * sigma).ceil() as isize;
    let w: Vec<f32> = (-r..=r).map(|d| (-((d * d) as f32) / (2.0 * sigma * sigma)).exp()).collect();
    let total: f32 = w.iter().sum();
    let pass = |src: &Image<f32>, along_x: bool| {
        src.map_xy(|x, y, _| {
            let mut s = 0.0;
            for d in -r..=r {
                let (xx, yy) = if along_x { (x as isize + d, y as isize) } else { (x as isize, y as isize + d) };
                s += w[(d + r) as usize] * src.get_border(xx, yy, Border::Mirror);
            }
            s / total
        })
    };
    pass(&pass(img, true), false)
}

/// PSNR in dB with peak 1 (= 255 grey levels).
fn psnr(a: &Image<f32>, truth: &Image<f32>) -> f32 {
    let mse: f32 = a.pixels().iter().zip(truth.pixels()).map(|(p, q)| (p - q).powi(2)).sum::<f32>() / a.pixels().len() as f32;
    10.0 * (1.0 / mse).log10()
}

/// The PSNR on one quarter of the image (x0, y0 = its top-left corner).
fn psnr_quarter(a: &Image<f32>, truth: &Image<f32>, x0: usize, y0: usize) -> f32 {
    psnr(&a.crop(x0, y0, 64, 64), &truth.crop(x0, y0, 64, 64))
}

pub fn main() {
    let dir = "out/denoise";
    let truth = test_image();
    let noisy = samples::add_noise(&truth, 20.0 / 255.0, 1);
    save_gray(&truth, format!("{dir}/truth.png"));
    save_gray(&noisy, format!("{dir}/noisy.png"));
    println!("test image, Gaussian noise σn = 20 grey levels; PSNR (peak 255) in dB");
    println!("  method                          all    step   ramp+line  lattice  spot   line pixel (truth {:.0})  spot centre (truth {:.0})", 255.0 * truth[(96, 32)], 255.0 * truth[(96, 96)]);
    let results = [
        ("noisy", noisy.clone()),
        ("Gaussian σ = 1.5", gaussian(&noisy, 1.5)),
        ("median 3 × 3", median_filter(&noisy, 1)),
        ("median 5 × 5", median_filter(&noisy, 2)),
        ("bilateral σs 2, σr 30", bilateral(&noisy, 4, 2.0, 30.0 / 255.0)),
        ("non-local means 3 × 3, 15 × 15", nonlocal_means(&noisy, 1, 7, 30.0 / 255.0)),
    ];
    for (name, img) in &results {
        println!(
            "  {name:30} {:5.1}   {:5.1}   {:5.1}      {:5.1}    {:5.1}  {:5.0}                {:5.0}",
            psnr(img, &truth),
            psnr_quarter(img, &truth, 0, 0),
            psnr_quarter(img, &truth, 64, 0),
            psnr_quarter(img, &truth, 0, 64),
            psnr_quarter(img, &truth, 64, 64),
            255.0 * img[(96, 32)],
            255.0 * img[(96, 96)]
        );
    }
    for (file, (_, img)) in ["noisy", "gaussian", "median3", "median5", "bilateral", "nlm"].iter().zip(&results).skip(1) {
        save_gray(img, format!("{dir}/{file}.png"));
        // the method noise: what the filter removed (0 = grey)
        save_gray_signed(&noisy.zip_map(img, |a, b| a - b), format!("{dir}/{file}-method-noise.png"));
    }

    // Salt and pepper: 10 % of the pixels set to 0 or 255.
    let mut rng = Rng::new(4);
    let salty = Image::from_fn(128, 128, |x, y| {
        let u = rng.uniform();
        if u < 0.05 { 0.0 } else if u < 0.10 { 1.0 } else { truth[(x, y)] }
    });
    println!("\nsalt and pepper (10 % of the pixels): noisy {:.1} dB, Gaussian σ = 1 {:.1} dB, median 3 × 3 {:.1} dB", psnr(&salty, &truth), psnr(&gaussian(&salty, 1.0), &truth), psnr(&median_filter(&salty, 1), &truth));
    save_gray(&salty, format!("{dir}/salt-pepper.png"));
    save_gray(&gaussian(&salty, 1.0), format!("{dir}/salt-pepper-gaussian.png"));
    save_gray(&median_filter(&salty, 1), format!("{dir}/salt-pepper-median3.png"));
    println!("\nwrote the pictures to {dir}/");
}
