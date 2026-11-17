// Blurs the course micrograph (Gaussian PSF, σ = 2 pixels), adds 1 % noise, and restores it
// with your inverse, truncated inverse, Wiener and Tikhonov filters. Results in out/deconv/.
//   cargo run -p day05 -- deconv                      (your code)
//   cargo run -p day05 --features solution -- deconv  (the model solution)

use crate::deconv::*;
use crate::common::rmse;
use imgcourse::*;

fn negative(a: &Image<f32>) -> f32 {
    100.0 * a.pixels().iter().filter(|&&v| v < 0.0).count() as f32 / a.pixels().len() as f32
}

pub fn main() {
    let dir = "out/deconv";
    let f = samples::cell();
    let (w, h) = (f.width(), f.height());
    let otf = gaussian_otf(2.0, w, h);
    let g = samples::add_noise(&apply_gain(&f, &otf), 0.01, 1);
    save_gray(&f, format!("{dir}/object.png"));
    save_gray(&g, format!("{dir}/observed.png"));
    println!("micrograph, Gaussian blur σ = 2 pixels, noise 0.01 (1 % of white); mean of the object {:.4}", f.mean());
    let true_snr = f.pixels().iter().map(|v| v * v).sum::<f32>() / f.pixels().len() as f32 / (0.01 * 0.01);
    println!("the image's own SNR, mean(object²) / σn²: {true_snr:.0}");
    println!("  observed                     RMSE {:.4}", rmse(&g, &f));

    let report = |name: &str, m: Method| {
        let x = restore(&g, &otf, m);
        println!("  {name:28} RMSE {:.4}   mean {:.4}   negative pixels {:4.1} %", rmse(&x, &f), x.mean(), negative(&x));
        x
    };
    let inverse = report("inverse", Method::Inverse);
    save_gray(&inverse, format!("{dir}/inverse.png"));
    for eps in [0.05, 0.15, 0.3] {
        let x = report(&format!("truncated, ε = {eps}"), Method::Truncated(eps));
        if eps == 0.15 {
            save_gray(&x, format!("{dir}/truncated.png"));
        }
    }
    println!();
    for snr in [3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 10000.0] {
        let x = report(&format!("Wiener, SNR {snr}"), Method::Wiener(snr));
        match snr as u32 {
            300 => save_gray(&x, format!("{dir}/wiener-snr300.png")),
            10000 => save_gray(&x, format!("{dir}/wiener-snr10000.png")),
            _ => {}
        }
    }
    println!();
    for lambda in [0.1, 0.01, 0.001] {
        report(&format!("Tikhonov energy, λ = {lambda}"), Method::TikhonovEnergy(lambda));
    }
    for lambda in [0.1, 0.01, 0.003, 0.001, 0.0003, 0.0001] {
        let x = report(&format!("Tikhonov roughness, λ = {lambda}"), Method::TikhonovRoughness(lambda));
        if lambda == 0.001 {
            save_gray(&x, format!("{dir}/tikhonov-roughness.png"));
        }
    }
    println!("\nwrote the pictures to {dir}/");
}
