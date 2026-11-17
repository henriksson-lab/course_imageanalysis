// Projections and sinograms of point masses, the Shepp–Logan phantom, a shifted disc and the
// micrograph. Pictures go to out/radon/ (sinograms: θ across, ρ up).
//   cargo run -p radon -- radon                      (your code)
//   cargo run -p radon --features solution -- radon  (the model solution)

use imgcourse::*;
use crate::common::{angles, centre, shepp_logan, sinogram_picture};
use crate::radon::*;
use std::f32::consts::PI;

pub fn main() {
    let dir = "out/radon";
    let n = 128;
    let c = centre(n);

    // 1. The three points of the lecture, 16 pixels per unit: A = (2, 0) mass 1, B = (0, 1)
    // mass 2, C = (−1, −1) mass 1. Each point mass is a tiny Gaussian spot (standard deviation
    // 1 pixel) holding that mass: a single bright pixel is too sharp to read between pixels.
    let three = [(2.0f32, 0.0f32, 1.0f32), (0.0, 1.0, 2.0), (-1.0, -1.0, 1.0)];
    let points = Image::from_fn(n, n, |px, py| {
        let (x, y) = (px as f32 - c, c - py as f32);
        let mut v = 0.0;
        for (u, w, m) in three {
            let d2 = (x - 16.0 * u).powi(2) + (y - 16.0 * w).powi(2);
            v += m * (-d2 / 2.0).exp() / (2.0 * PI);
        }
        v
    });
    save_gray_stretched(&points, format!("{dir}/points.png"));
    for deg in [0.0f32, 45.0, 90.0] {
        let p = projection(&points, deg.to_radians());
        // for each point: where C1 says it lands, and the mass the projection holds near there
        let bars: Vec<String> = three
            .iter()
            .map(|&(u, w, _)| {
                let r = (rho(u, w, deg.to_radians()) * 100.0).round() / 100.0 + 0.0; // no "−0.00"
                let k = (16.0 * r + c).round() as usize;
                let mass: f32 = p[k - 4..=k + 4].iter().sum();
                format!("ρ = {r:5.2}: {mass:.2}")
            })
            .collect();
        println!("three points at {deg:>2}°: projection sum {:.2};  {}", p.iter().sum::<f32>(), bars.join(",  "));
    }
    let a = rho(2.0, 0.0, 2.0f32.atan());
    println!("A and B on one ray at θ = {:.1}°, ρ = {a:.2} units", 2.0f32.atan().to_degrees());
    let thin = sinogram(&points, &angles(180));
    save_gray(&sinogram_picture(&thin), format!("{dir}/points-sinogram.png"));

    // 2. The Shepp–Logan phantom and its sinogram.
    let phantom = shepp_logan(n);
    save_gray(&phantom, format!("{dir}/phantom.png"));
    let sino = sinogram(&phantom, &angles(180));
    save_gray(&sinogram_picture(&sino), format!("{dir}/phantom-sinogram.png"));
    let sums: Vec<f32> = [0, 30, 45, 90].iter().map(|&a| (0..n).map(|k| sino[(a, k)]).sum()).collect();
    println!("phantom: total {:.1}; projection sums at 0°, 30°, 45°, 90°: {sums:.1?}", phantom.sum());

    // 3. A disc, centred and shifted by (20, 12) pixels (x right, y up), over a full turn.
    let full: Vec<f32> = (0..360).map(|d| (d as f32).to_radians()).collect();
    for (name, dx, dy) in [("disc-centred", 0.0, 0.0), ("disc-shifted", 20.0, 12.0)] {
        let disc = Image::from_fn(n, n, |x, y| {
            let (px, py) = (x as f32 - c - dx, c - y as f32 - dy);
            if px * px + py * py <= 15.0 * 15.0 { 1.0 } else { 0.0 }
        });
        save_gray(&disc, format!("{dir}/{name}.png"));
        save_gray(&sinogram_picture(&sinogram(&disc, &full)), format!("{dir}/{name}-sinogram-360.png"));
    }
    println!("the shifted disc's band follows ρ = 20 cos θ + 12 sin θ: largest {:.1} at θ = {:.0}°", (20.0f32 * 20.0 + 12.0 * 12.0).sqrt(), 12.0f32.atan2(20.0) * 180.0 / PI);

    // 4. The micrograph (only the inscribed circle is seen from every angle).
    let cell = samples::cell().crop(64, 64, n, n);
    save_gray(&cell, format!("{dir}/cell.png"));
    save_gray(&sinogram_picture(&sinogram(&cell, &angles(180))), format!("{dir}/cell-sinogram.png"));
    println!("wrote pictures to {dir}/");
}
