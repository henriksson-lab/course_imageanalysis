// Self-supervised denoising of a micrograph crop with a linear filter: Noise2Void (blind spot,
// trained on the noisy image alone), the same without the blind spot, Noise2Noise (a second
// noisy image as the target) and a supervised filter (clean target), next to a Gaussian blur
// and non-local means. Once with independent noise, once with correlated noise.
// Pictures in out/self-denoise/.
//   cargo run -p selfsup -- self-denoise                      (your code)
//   cargo run -p selfsup --features solution -- self-denoise  (the model solution)

use crate::common::{enlarge, noise, nonlocal_means, side_by_side};
use crate::self_denoise::*;
use imgcourse::{filters, samples, save_gray, Image};

fn add(a: &Image<f32>, b: &Image<f32>) -> Image<f32> {
    a.zip_map(b, |p, q| p + q)
}

pub fn main() {
    let dir = "out/self-denoise";
    let clean = samples::cell().crop(64, 64, 128, 128);
    let sigma = 0.1;
    let radius = 2;
    for (case, corr) in [("independent", 0.0), ("correlated", 1.0)] {
        let noisy = add(&clean, &noise(128, 128, sigma, corr, 11));
        let noisy2 = add(&clean, &noise(128, 128, sigma, corr, 12));
        println!("{case} noise, standard deviation {sigma} (PSNR in dB against the clean crop, peak 1)");
        let psnr = |img: &Image<f32>| filters::psnr(img, &clean, 1.0);
        println!("  {:<44} {:>6.2}", "noisy image", psnr(&noisy));

        let mut best = (0.0, 0.0);
        for s in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0] {
            let p = psnr(&filters::gaussian_blur(&noisy, s));
            if p > best.1 {
                best = (s, p);
            }
        }
        println!("  {:<44} {:>6.2}", format!("Gaussian blur, best sigma {} (chosen with the truth)", best.0), best.1);
        let nlm = nonlocal_means(&noisy, 1, 5, sigma);
        println!("  {:<44} {:>6.2}", "non-local means (3 x 3, search 11 x 11, h = 0.1)", psnr(&nlm));

        let n2v = fit_filter(&noisy, &noisy, radius, true);
        let identity = fit_filter(&noisy, &noisy, radius, false);
        let n2n = fit_filter(&noisy, &noisy2, radius, false);
        let sup = fit_filter(&noisy, &clean, radius, false);
        let sup_blind = fit_filter(&noisy, &clean, radius, true);
        let mut outs = Vec::new();
        for (name, f) in [
            ("Noise2Void: blind spot, noisy target", &n2v),
            ("no blind spot, noisy target", &identity),
            ("Noise2Noise: second noisy image as target", &n2n),
            ("supervised blind spot: clean target", &sup_blind),
            ("supervised: clean target", &sup),
        ] {
            let out = apply_filter(&noisy, f);
            println!("  {name:<44} {:>6.2}   centre weight {:>5.2}", psnr(&out), f.weight(0, 0));
            outs.push(out);
        }
        let (mixed, c) = mix_back(&noisy, &outs[0], sigma);
        println!("  {:<44} {:>6.2}   c = {c:.2}", "Noise2Void, own pixel mixed back (sigma known)", psnr(&mixed));
        println!("  Noise2Void weights (5 x 5, centre = the blind spot), bias {:.3}:", n2v.bias);
        for dy in -2..=2 {
            let row: Vec<String> = (-2..=2).map(|dx| format!("{:>6.3}", n2v.weight(dx, dy))).collect();
            println!("    {}", row.join(" "));
        }
        let sum: f32 = n2v.weights.iter().sum();
        println!("  sum of the Noise2Void weights {sum:.3}\n");
        save_gray(&enlarge(&side_by_side(&[&clean, &noisy, &outs[0], &nlm]), 2), format!("{dir}/{case}-clean-noisy-n2v-nlm.png"));
        save_gray(&enlarge(&outs[1], 2), format!("{dir}/{case}-no-blind-spot.png"));
    }
    println!("wrote the pictures to {dir}/");
}
