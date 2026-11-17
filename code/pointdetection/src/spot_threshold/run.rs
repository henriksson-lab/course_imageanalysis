// How many false spots does a threshold let through? Counts noise maxima in seeded noise
// images and compares with the Euler formula; then Poisson noise with a global threshold, a
// local noise estimate and the Anscombe transform. Pictures in out/spot-threshold/.
//   cargo run -p pointdetection -- spot-threshold                      (your code)
//   cargo run -p pointdetection --features solution -- spot-threshold  (the model solution)

use imgcourse::*;
use crate::spot_threshold::sim::{draw_marks, filter_noise_factor, gaussian_blur, noise_image, normal_tail};
use crate::spot_threshold::*;

pub fn main() {
    let dir = "out/spot-threshold";
    // 1. White noise (sd 10 grey levels), Gaussian filter σ_f = 1.5, maxima in 3 × 3 windows.
    let (size, n_images, sigma_f) = (512, 12, 1.5);
    let ks = [3.0, 4.0, 5.0];
    let (mut maxima, mut above) = (0usize, [0usize; 3]);
    let mut estimates = Vec::new();
    for seed in 0..n_images {
        let img = noise_image(size, 0.4, 10.0 / 255.0, seed);
        let sigma = estimate_noise(&img);
        estimates.push(sigma * 255.0);
        let z = gaussian_blur(&img, sigma_f).map(|v| (v - 0.4) / (sigma * filter_noise_factor(sigma_f, 0.0)));
        maxima += maxima_above(&z, f32::NEG_INFINITY).len();
        for (i, &k) in ks.iter().enumerate() {
            above[i] += maxima_above(&z, k).len();
        }
        if seed == 0 {
            save_rgb(&draw_marks(&z, &maxima_above(&z, 3.0)), format!("{dir}/noise-k3.png"));
        }
    }
    let per = |n: usize| n as f32 / n_images as f32;
    println!("{n_images} noise images of {size} × {size}, noise sd 10, filter σ_f = {sigma_f}");
    println!("noise estimated from the first three images: {:.2} {:.2} {:.2}", estimates[0], estimates[1], estimates[2]);
    println!("local maxima per image: {:.0}", per(maxima));
    println!("   k   false spots per image   Euler formula   naive (maxima × P(Z > k))");
    for (i, &k) in ks.iter().enumerate() {
        let euler = euler_expected((size * size) as f32, sigma_f, k);
        println!("{k:>4.0}   {:>20.1}   {euler:>13.2}   {:>10.2}", per(above[i]), per(maxima) * normal_tail(k));
    }

    // 2. Poisson noise: 20 photons per pixel, a broad bright patch of up to 200, no spots.
    // Filter: Gaussian σ = 1.5 minus Gaussian σ = 4 (zero-mean), k = 4. Every detection is false.
    let n = 256;
    let mut rng = Rng::new(3);
    let mean = Image::from_fn(n, n, |x, y| {
        let d2 = (x as f32 - 170.0).powi(2) + (y as f32 - 110.0).powi(2);
        20.0 + 180.0 * (-d2 / (2.0 * 50.0f32.powi(2))).exp()
    });
    let counts = Image::from_fn(n, n, |x, y| rng.poisson(mean[(x, y)]));
    save_gray_stretched(&counts, format!("{dir}/poisson.png"));
    let factor = filter_noise_factor(1.5, 4.0);
    let response = |img: &Image<f32>| gaussian_blur(img, 1.5).zip_map(&gaussian_blur(img, 4.0), |a, b| a - b);
    let on_patch = |marks: &[(usize, usize)]| marks.iter().filter(|&&(x, y)| mean[(x, y)] > 60.0).count();

    let sigma = estimate_noise(&counts);
    let z_global = response(&counts).map(|v| v / (sigma * factor));
    let global = maxima_above(&z_global, 4.0);
    let background = gaussian_blur(&counts, 8.0);
    let z_local = response(&counts).zip_map(&background, |v, b| v / (b.max(1.0).sqrt() * factor));
    let local = maxima_above(&z_local, 4.0);
    let a = anscombe(&counts);
    let sigma_a = estimate_noise(&a);
    let z_ansc = response(&a).map(|v| v / (sigma_a * factor));
    let ansc = maxima_above(&z_ansc, 4.0);
    println!("Poisson image (20 to 200 photons), k = 4; false spots (of them on the bright patch):");
    println!("  global noise estimate {sigma:.2}: {} ({})", global.len(), on_patch(&global));
    println!("  local noise sqrt(background):  {} ({})", local.len(), on_patch(&local));
    println!("  Anscombe, estimate {sigma_a:.2}:    {} ({})", ansc.len(), on_patch(&ansc));
    save_rgb(&draw_marks(&z_global, &global), format!("{dir}/poisson-global.png"));
    save_rgb(&draw_marks(&z_local, &local), format!("{dir}/poisson-local.png"));
    save_rgb(&draw_marks(&z_ansc, &ansc), format!("{dir}/poisson-anscombe.png"));
    println!("wrote the pictures to {dir}/ (response in noise sd, red = above the threshold)");
}
