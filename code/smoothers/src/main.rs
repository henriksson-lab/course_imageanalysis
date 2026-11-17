// Compares smoothers on a noisy test card with known truth; results in out/smoothers/.
//   cargo run -p smoothers                      (your code)
//   cargo run -p smoothers --features solution  (the model solution)

use imgcourse::*;
use smoothers::*;

/// A test card with known truth: a dark background (40), a bright square (200), discs of
/// radius 2 to 8, a dot lattice of period 6, fine stripes, a ramp and a 1-pixel line
/// (grey values / 255). The line is at x = 120, rows 132-151.
fn test_card(size: usize) -> Image<f32> {
    Image::from_fn(size, size, |x, y| {
        let (xf, yf) = (x as f32, y as f32);
        let mut v = 40.0;
        if (16..64).contains(&x) && (16..64).contains(&y) {
            v = 200.0;
        }
        for (i, r) in [2.0, 3.0, 5.0, 8.0].iter().enumerate() {
            let (cx, cy) = (84.0 + 20.0 * i as f32, 24.0);
            if (xf - cx).powi(2) + (yf - cy).powi(2) <= r * r {
                v = 160.0;
            }
        }
        if (16..64).contains(&x) && (76..124).contains(&y) && x % 6 < 2 && y % 6 < 2 {
            v = 180.0; // a lattice of 2 × 2 dots
        }
        if (76..148).contains(&x) && (76..124).contains(&y) && x % 4 < 2 {
            v = 120.0; // stripes of period 4
        }
        if (132..152).contains(&y) {
            v = if x < 16 { 40.0 } else if x < 96 { 40.0 + 2.0 * (x - 16) as f32 } else { 200.0 };
            if x == 120 {
                v = 40.0; // a 1-pixel line
            }
        }
        v / 255.0
    })
}

/// How much of the 1-pixel line's contrast is kept (1 = all), averaged over its rows.
fn line_kept(u: &Image<f32>, truth: &Image<f32>) -> f32 {
    let depth = |img: &Image<f32>| (136..148).map(|y| 0.5 * (img[(117, y)] + img[(123, y)]) - img[(120, y)]).sum::<f32>();
    depth(u) / depth(truth)
}

fn main() {
    let dir = "out/smoothers";
    let truth = test_card(160);
    let noisy = samples::add_noise(&truth, 15.0 / 255.0, 1);
    save_gray(&truth, format!("{dir}/truth.png"));
    save_gray(&noisy, format!("{dir}/noisy.png"));
    println!("noisy (15 grey levels): {:.1} dB", psnr(&noisy, &truth));

    // Each method at its best PSNR over a few settings.
    let best = |name: &str, settings: &[f32], f: &dyn Fn(f32) -> Image<f32>| {
        let mut b = (f32::NEG_INFINITY, 0.0, noisy.clone());
        for &s in settings {
            let out = f(s);
            let p = psnr(&out, &truth);
            if p > b.0 {
                b = (p, s, out);
            }
        }
        let (p, s, out) = b;
        println!("{name:<18} best setting {s:5.2}: {p:.1} dB, line kept {:.2}", line_kept(&out, &truth));
        save_gray(&out, format!("{dir}/{name}.png"));
        // method noise: what the smoother removed (0 = grey)
        save_gray(&noisy.zip_map(&out, |a, b| 0.5 + 2.0 * (a - b)), format!("{dir}/{name}-method-noise.png"));
    };
    best("gaussian", &[0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5], &|s| gaussian_blur(&noisy, s));
    best("median", &[1.0, 2.0], &|r| median_filter(&noisy, r as usize));
    best("haar-shrinkage", &[5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0], &|t| haar_shrink(&noisy, t / 255.0));

    // Salt and pepper: 5 % of the pixels set to 0 or 1.
    let mut rng = Rng::new(2);
    let salt = truth.map(|v| {
        let r = rng.uniform();
        if r < 0.025 { 0.0 } else if r < 0.05 { 1.0 } else { v }
    });
    println!(
        "salt and pepper: noisy {:.1} dB, median radius 1 {:.1} dB, Gaussian 0.7 {:.1} dB",
        psnr(&salt, &truth),
        psnr(&median_filter(&salt, 1), &truth),
        psnr(&gaussian_blur(&salt, 0.7), &truth)
    );
    save_gray(&median_filter(&salt, 1), format!("{dir}/salt-median.png"));

    // Smoothing before thresholding at 120 (noise 40 grey levels).
    let strong = samples::add_noise(&truth, 40.0 / 255.0, 3);
    let level = 120.0 / 255.0;
    println!("threshold at 120, noise 40, wrong pixels:");
    println!("  none            {:5}", threshold_errors(&strong, &truth, level));
    println!("  Gaussian 2      {:5}", threshold_errors(&gaussian_blur(&strong, 2.0), &truth, level));
    println!("  median radius 2 {:5}", threshold_errors(&median_filter(&strong, 2), &truth, level));
    println!("wrote the pictures to {dir}/");
}
