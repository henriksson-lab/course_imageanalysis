// Runs the structure tensor on test images and on the course micrograph; results in
// out/structure-tensor/.
//   cargo run -p edges -- structure-tensor                      (your code)
//   cargo run -p edges --features solution -- structure-tensor  (the model solution)

use imgcourse::*;
use std::f32::consts::PI;
use crate::structure_tensor::*;

/// Hue (0 … 1, once around the colour wheel), saturation and value to [red, green, blue].
fn hsv(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h6 = (h.rem_euclid(1.0)) * 6.0;
    let f = h6.fract();
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f)));
    match h6 as u32 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// The orientation map: hue = orientation (an axis, so 180° is once around the wheel),
/// saturation = coherence, brightness = energy μ1 + μ2 (relative to the largest).
fn orientation_map(j: &Image<[f32; 3]>) -> Image<[f32; 3]> {
    let eig = j.map(eigen_tensor);
    let top = eig.pixels().iter().fold(0.0f32, |m, e| m.max(e.0 + e.1));
    eig.map(|(mu1, mu2, theta)| hsv(theta / PI, coherence(mu1, mu2), ((mu1 + mu2) / (0.5 * top)).sqrt().min(1.0)))
}

/// Mark each corner with a small red cross.
fn mark(img: &Image<f32>, points: &[(usize, usize)]) -> Image<[f32; 3]> {
    let mut out = gray_to_rgb(img);
    for &(x, y) in points {
        for d in -3..=3isize {
            for (px, py) in [(x as isize + d, y as isize), (x as isize, y as isize + d)] {
                if out.contains(px, py) {
                    out[(px as usize, py as usize)] = [1.0, 0.1, 0.1];
                }
            }
        }
    }
    out
}

pub fn main() {
    let dir = "out/structure-tensor";
    let mut rng = Rng::new(11);

    // A bright square: edge, corner and flat, σ = 1, ρ = 3 (grey levels 0 … 255).
    let square = Image::from_fn(128, 128, |x, y| if (32..96).contains(&x) && (32..96).contains(&y) { 200.0 } else { 40.0 });
    let j = structure_tensor(&square, 1.0, 3.0);
    for (name, x, y) in [("edge", 32, 64), ("corner", 32, 32), ("flat", 64, 64)] {
        // Round away the tiny rounding errors (about 1e-10) of the flat inside, so they print as 0.
        let t = j[(x, y)].map(|v| if v.abs() < 1e-3 { 0.0 } else { v });
        let (mu1, mu2, theta) = eigen_tensor(t);
        println!(
            "square, {name:6} ({x}, {y}): J = [{:.0} {:.0}; {:.0} {:.0}], μ1 = {mu1:.0}, μ2 = {mu2:.0}, coherence {:.2}, gradient at {:.0}°, Harris {:.0}",
            t[0],
            t[1],
            t[1],
            t[2],
            coherence(mu1, mu2),
            theta.to_degrees(),
            harris(t, 0.05)
        );
    }

    // Stripes at 30°, with noise: averaging gradients against averaging g gᵀ.
    let (c, s) = ((30f32).to_radians().cos(), (30f32).to_radians().sin());
    let stripes = Image::from_fn(128, 128, |x, y| 128.0 + 80.0 * (2.0 * PI * (c * x as f32 - s * y as f32) / 8.0).sin() + 15.0 * rng.normal());
    save_gray(&stripes.map(|v| v / 255.0), format!("{dir}/stripes.png"));
    let (lx, ly) = gradient(&stripes, 1.0);
    let (mx, my) = (gaussian_blur(&lx, 3.0), gaussian_blur(&ly, 3.0));
    let j = structure_tensor(&stripes, 1.0, 3.0);
    let (mu1, mu2, theta) = eigen_tensor(j[(64, 64)]);
    println!(
        "stripes at 30°, (64, 64): mean gradient length {:.2}, √μ1 = {:.1}, coherence {:.2}, structure at {:.0}° counter-clockwise on the screen",
        mx[(64, 64)].hypot(my[(64, 64)]),
        mu1.sqrt(),
        coherence(mu1, mu2),
        (90.0 - theta.to_degrees()).rem_euclid(180.0)
    );
    // The naive map uses the angle and length of the mean gradient instead.
    let top = mx.zip_map(&my, |a, b| a.hypot(b)).max().max(mu1.sqrt());
    let naive = mx.zip_map(&my, |a, b| hsv(b.atan2(a) / PI, 1.0, (a.hypot(b) / top).min(1.0)));
    save_rgb(&naive, format!("{dir}/stripes-mean-gradient.png"));
    save_rgb(&orientation_map(&j), format!("{dir}/stripes-orientation.png"));

    // Concentric rings next to an isotropic blotchy texture.
    let blotches = gaussian_blur(&Image::from_fn(256, 128, |_, _| rng.normal()), 2.0);
    let scene = Image::from_fn(256, 128, |x, y| {
        if x < 128 {
            let r = (x as f32 - 64.0).hypot(y as f32 - 64.0);
            128.0 + 80.0 * (2.0 * PI * r / 10.0).sin()
        } else {
            128.0 + 400.0 * blotches[(x, y)]
        }
    });
    let scene = samples::add_noise(&scene, 10.0, 12);
    save_gray(&scene.map(|v| v / 255.0), format!("{dir}/rings.png"));
    let j = structure_tensor(&scene, 1.0, 3.0);
    save_rgb(&orientation_map(&j), format!("{dir}/rings-orientation.png"));
    let coh = |x: usize, y: usize| {
        let (a, b, _) = eigen_tensor(j[(x, y)]);
        coherence(a, b)
    };
    println!("rings and texture: coherence {:.2} on the rings (64, 30), {:.2} at the ring centre, {:.2} on the texture (192, 64)", coh(64, 30), coh(64, 64), coh(192, 64));

    // Harris corners on shapes: a square, a triangle, a disc and a checkerboard.
    let shapes = Image::from_fn(256, 128, |x, y| {
        let (xf, yf) = (x as f32, y as f32);
        let sq = (16..56).contains(&x) && (16..56).contains(&y);
        let tri = yf > 70.0 && yf < 112.0 && (xf - 36.0).abs() < 0.5 * (yf - 70.0);
        let disc = (xf - 100.0).hypot(yf - 64.0) < 22.0;
        let check = (144..240).contains(&x) && (16..112).contains(&y) && ((x - 144) / 16 + (y - 16) / 16) % 2 == 0;
        if sq || tri || disc || check { 200.0 } else { 40.0 }
    });
    let j = structure_tensor(&shapes, 1.0, 2.0);
    let r = j.map(|t| harris(t, 0.05));
    let found = corners(&r, 0.05 * r.max());
    println!("shapes: {} corners", found.len());
    save_rgb(&mark(&shapes.map(|v| v / 255.0), &found), format!("{dir}/corners.png"));

    // The micrograph: orientation map and corners.
    let cell = samples::cell().map(|v| 255.0 * v);
    let j = structure_tensor(&cell, 1.5, 4.0);
    save_rgb(&orientation_map(&j), format!("{dir}/cell-orientation.png"));
    let coh = |x: usize, y: usize| {
        let (a, b, _) = eigen_tensor(j[(x, y)]);
        coherence(a, b)
    };
    println!("micrograph, σ = 1.5, ρ = 4: coherence {:.2} on the nuclear envelope (94, 126), {:.2} inside the nucleus (150, 150)", coh(94, 126), coh(150, 150));
    println!("wrote the results to {dir}/");
}
