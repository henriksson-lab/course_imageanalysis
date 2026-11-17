// Runs your heat flow on the micrograph and on a row of numbers; results in out/heat-equation/.
//   cargo run -p day10 -- heat-equation                      (your code)
//   cargo run -p day10 --features solution -- heat-equation  (the model solution)

use crate::heat_equation::*;
use imgcourse::*;

pub fn main() {
    let dir = "out/heat-equation";
    let img = samples::cell();
    save_gray(&img, format!("{dir}/input.png"));

    // Heat flow is a Gaussian blur: 10, 40, 160 steps of 0.2 against sigma = sqrt(2 t).
    println!("heat flow against a Gaussian blur (dt = 0.2):");
    for steps in [10, 40, 160] {
        let t = 0.2 * steps as f32;
        let sigma = (2.0 * t).sqrt();
        let heat = heat_flow(&img, 0.2, steps, Border::Mirror);
        let gauss = gaussian_blur(&img, sigma);
        let diff = heat.zip_map(&gauss, |a, b| a - b);
        let largest = diff.pixels().iter().fold(0.0f32, |m, v| m.max(v.abs()));
        println!("  {steps:3} steps, t = {t:4.0}, sigma = {sigma}: largest difference {:.2} grey levels", largest * 255.0);
        save_gray(&heat, format!("{dir}/heat-{steps}-steps.png"));
        save_gray(&gauss, format!("{dir}/gauss-sigma-{sigma}.png"));
    }

    // Too large a step: 1.2 times the 2D limit 1/4 on the noisy micrograph.
    let noisy = samples::add_noise(&img, 10.0 / 255.0, 1);
    println!("dt = 0.3 (1.2 times the limit), noisy micrograph:");
    let mut u = noisy.clone();
    for step in 1..=24 {
        u = heat_step(&u, 0.3, Border::Mirror);
        if step % 4 == 0 {
            println!("  after {step:2} steps: values from {:9.2} to {:9.2}", u.min(), u.max());
            save_gray(&u, format!("{dir}/unstable-{step}-steps.png"));
        }
    }

    // Zero against mirror border: the total after 150 steps.
    let small = img.crop(96, 96, 64, 64);
    for (name, border) in [("mirror", Border::Mirror), ("zero", Border::Zero)] {
        let after = heat_flow(&small, 0.2, 150, border);
        println!("{name} border, 64 × 64 crop: total {:.1} before, {:.1} after 150 steps", small.sum(), after.sum());
    }

    // A step edge, three steps of dt = 2: explicit against implicit.
    let mut explicit: Vec<f32> = (0..12).map(|i| if i < 6 { 0.0 } else { 1.0 }).collect();
    let mut implicit = explicit.clone();
    for _ in 0..3 {
        explicit = heat_step_1d(&explicit, 2.0);
        implicit = implicit_step_1d(&implicit, 2.0);
    }
    let show = |v: &[f32]| v.iter().map(|x| format!("{x:.2}")).collect::<Vec<_>>().join(" ");
    println!("step edge after 3 steps of dt = 2:");
    println!("  explicit: {}", show(&explicit));
    println!("  implicit: {}", show(&implicit));
    println!("wrote the pictures to {dir}/");
}
