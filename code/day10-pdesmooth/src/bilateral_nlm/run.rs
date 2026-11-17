// Runs your bilateral filter and non-local means on a noisy test card; results in
// out/bilateral-nlm/.
//   cargo run -p day10 -- bilateral-nlm                      (your code)
//   cargo run -p day10 --features solution -- bilateral-nlm  (the model solution)

use crate::bilateral_nlm::*;
use crate::common::test_card;
use imgcourse::*;

fn psnr(a: &Image<f32>, truth: &Image<f32>) -> f32 {
    let mse = a.zip_map(truth, |p, q| (p - q) * (p - q)).mean();
    10.0 * (1.0 / mse).log10()
}

pub fn main() {
    let dir = "out/bilateral-nlm";
    let truth = test_card(160);
    let sn = 10.0 / 255.0; // noise of 10 grey levels
    let noisy = samples::add_noise(&truth, sn, 1);
    save_gray(&truth, format!("{dir}/truth.png"));
    save_gray(&noisy, format!("{dir}/noisy.png"));
    println!("noisy: {:.1} dB", psnr(&noisy, &truth));

    // The bilateral filter, sigma_s = 2 (window 9 × 9), three range parameters.
    for sr in [5.0, 20.0, 80.0] {
        let out = bilateral(&noisy, 2.0, sr / 255.0, 4);
        println!("bilateral, sigma_r = {sr:2}: {:.1} dB", psnr(&out, &truth));
        save_gray(&out, format!("{dir}/bilateral-sr{sr}.png"));
    }
    // Iterated: 1, 3 and 10 passes with sigma_r = 30.
    let mut u = noisy.clone();
    for pass in 1..=10 {
        u = bilateral(&u, 2.0, 30.0 / 255.0, 4);
        if [1, 3, 10].contains(&pass) {
            println!("bilateral, sigma_r = 30, {pass:2} passes: {:.1} dB", psnr(&u, &truth));
            save_gray(&u, format!("{dir}/bilateral-{pass}-passes.png"));
        }
    }

    // Non-local means: 3 × 3 patches, 15 × 15 search window, h = 1.5 sigma_n.
    let out = nlm(&noisy, 1, 7, 1.5 * sn);
    println!("non-local means, h = 15: {:.1} dB", psnr(&out, &truth));
    save_gray(&out, format!("{dir}/nlm.png"));

    // N_eff of NLM at a few pixels: a lattice dot, the flat background, the square's corner.
    for (name, (x, y)) in [("lattice dot", (30, 78)), ("flat background", (8, 8)), ("corner of the square", (16, 16))] {
        let (px, py) = (x as isize, y as isize);
        let mut weights = Vec::new();
        let mut values = Vec::new();
        for qy in py - 7..=py + 7 {
            for qx in px - 7..=px + 7 {
                if !noisy.contains(qx, qy) {
                    continue;
                }
                let mut d2 = 0.0;
                for ky in -1..=1 {
                    for kx in -1..=1 {
                        let d = noisy.get_border(px + kx, py + ky, Border::Mirror) - noisy.get_border(qx + kx, qy + ky, Border::Mirror);
                        d2 += d * d;
                    }
                }
                weights.push((-(d2 / 9.0) / (1.5 * sn).powi(2)).exp());
                values.push(noisy[(qx as usize, qy as usize)]);
            }
        }
        let (m, n) = weighted_mean(&values, &weights);
        println!("NLM at the {name} ({x}, {y}): N_eff = {n:.1}, estimate {:.0}, truth {:.0}", 255.0 * m, 255.0 * truth[(x, y)]);
    }
    println!("wrote the pictures to {dir}/");
}
