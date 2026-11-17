// Smooths a noisy 1D profile (step, ramp, narrow peak) and a noisy micrograph with the
// quadratic prior and with total variation. Plots and pictures in out/gmrf/.
//   cargo run -p day28 -- gmrf                      (your code)
//   cargo run -p day28 --features solution -- gmrf  (the model solution)

use crate::gmrf::*;
use imgcourse::*;

/// The profile of the lecture: a step at 40 (0 → 2), a ramp from 2 down to 0 over 80–140,
/// a peak of height 2 at 166–169; 200 samples.
fn truth() -> Vec<f32> {
    (0..200)
        .map(|p| match p {
            0..=39 => 0.0,
            40..=79 => 2.0,
            80..=139 => 2.0 * (140 - p) as f32 / 60.0,
            166..=169 => 2.0,
            _ => 0.0,
        })
        .collect()
}

fn rmse(a: &[f32], b: &[f32]) -> f32 {
    (a.iter().zip(b).map(|(p, q)| (p - q).powi(2)).sum::<f32>() / a.len() as f32).sqrt()
}

/// A plot of the data (grey dots), the truth (green) and an estimate (orange), values −1 … 3.
fn plot(data: &[f32], truth: &[f32], estimate: &[f32], path: String) {
    let (w, h, sx) = (600, 220, 3);
    let row = |v: f32| ((3.0 - v) / 4.0 * (h - 1) as f32).round().clamp(0.0, (h - 1) as f32) as usize;
    let mut img = Image::new(w, h, [1.0f32; 3]);
    for x in 0..w {
        img[(x, row(0.0))] = [0.85, 0.85, 0.85]; // the zero line
    }
    for (p, &v) in data.iter().enumerate() {
        img[(p * sx + 1, row(v))] = [0.45, 0.45, 0.45];
    }
    for (curve, colour) in [(truth, [0.1, 0.6, 0.2]), (estimate, [0.9, 0.45, 0.0])] {
        for p in 0..curve.len() - 1 {
            let (r0, r1) = (row(curve[p]), row(curve[p + 1]));
            for x in p * sx..=(p + 1) * sx {
                let t = (x - p * sx) as f32 / sx as f32;
                img[(x + 1, (r0 as f32 + t * (r1 as f32 - r0 as f32)).round() as usize)] = colour;
            }
            for r in r0.min(r1)..=r0.max(r1) {
                img[((p + 1) * sx + 1, r)] = colour; // vertical part of a jump
            }
        }
    }
    save_rgb(&img, path);
}

pub fn main() {
    let dir = "out/gmrf";
    let truth = truth();
    let mut rng = Rng::new(1);
    let y: Vec<f32> = truth.iter().map(|v| v + 0.3 * rng.normal()).collect();
    println!("1D profile, noise σ = 0.3: RMSE of the noisy data {:.3}", rmse(&y, &truth));

    println!("\nquadratic:   λ    RMSE   peak height   energy (data + prior)");
    let mut best = (f32::INFINITY, 0.0);
    for lambda in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0] {
        let x = quadratic_1d(&y, lambda);
        let r = rmse(&x, &truth);
        if r < best.0 {
            best = (r, lambda);
        }
        let peak = x[166..170].iter().fold(f32::MIN, |m, &v| m.max(v));
        println!("        {lambda:6.1}  {r:.3}   {peak:.2}         {:.2}", energy_1d(&x, &y, lambda, Penalty::Quadratic));
    }
    println!("  best of these: λ = {} (RMSE {:.3})", best.1, best.0);
    plot(&y, &truth, &quadratic_1d(&y, 10.0), format!("{dir}/quadratic-10.png"));

    println!("\nTV:          λ    RMSE   peak height   energy");
    for lambda in [0.5, 1.0, 2.0, 3.0] {
        let x = tv_1d(&y, lambda, 20_000);
        let peak = x[166..170].iter().fold(f32::MIN, |m, &v| m.max(v));
        println!("        {lambda:6.1}  {:.3}   {peak:.2}         {:.2}", rmse(&x, &truth), energy_1d(&x, &y, lambda, Penalty::Tv));
        if lambda == 1.0 || lambda == 3.0 {
            plot(&y, &truth, &x, format!("{dir}/tv-{lambda}.png"));
        }
    }

    // 2D: the micrograph with noise, smoothed both ways.
    let img = samples::cell();
    let noisy = samples::add_noise(&img, 0.1, 2);
    let rmse2 = |a: &Image<f32>| (a.zip_map(&img, |p, q| (p - q).powi(2)).mean()).sqrt();
    save_gray(&noisy, format!("{dir}/noisy.png"));
    let quad = quadratic_2d(&noisy, 2.0, 100);
    let tv = tv_2d(&noisy, 0.08, 300);
    println!("\nmicrograph, noise σ = 0.1: RMSE noisy {:.4}, quadratic (λ = 2) {:.4}, TV (λ = 0.08) {:.4}", rmse2(&noisy), rmse2(&quad), rmse2(&tv));
    save_gray(&quad, format!("{dir}/quadratic-2d.png"));
    save_gray(&tv, format!("{dir}/tv-2d.png"));
    println!("\nwrote the plots and pictures to {dir}/");
}
