// Moments of test shapes and of the course micrograph; results in out/moments-basis/.
//   cargo run -p day18 -- moments-basis                      (your code)
//   cargo run -p day18 --features solution -- moments-basis  (the model solution)

use imgcourse::*;
use crate::moments_basis::*;
use std::time::Instant;

/// A 64 × 64 mask: 1 where `inside(x, y)` (pixel centres), 0 elsewhere.
fn mask(inside: impl Fn(f32, f32) -> bool) -> Image<f32> {
    Image::from_fn(64, 64, |x, y| if inside(x as f32, y as f32) { 1.0 } else { 0.0 })
}

pub fn main() {
    let dir = "out/moments-basis";
    let disc = |cx: f32, cy: f32, r: f32| move |x: f32, y: f32| (x - cx).hypot(y - cy) <= r;
    let pear = mask(|x, y| disc(24.0, 32.0, 13.0)(x, y) || disc(41.0, 32.0, 7.0)(x, y) || ((24.0..42.0).contains(&x) && (y - 32.0).abs() <= 5.0));
    let mirror = Image::from_fn(64, 64, |x, y| pear[(63 - x, y)]);
    let square = mask(|x, y| (17.0..47.0).contains(&x) && (17.0..47.0).contains(&y));
    let round = mask(disc(31.5, 31.5, 30.0 / 3f32.sqrt()));
    let ell = mask(|x, y| (12.0..22.0).contains(&x) && (10.0..54.0).contains(&y) || (12.0..50.0).contains(&x) && (44.0..54.0).contains(&y));
    // the pear in grey: brighter to the right
    let grey = pear.map_xy(|x, _, v| v * (0.3 + 0.7 * x as f32 / 63.0));

    let shapes = [("pear", &pear), ("mirror", &mirror), ("square", &square), ("disc", &round), ("L", &ell), ("grey pear", &grey)];
    println!("{:>10} {:>6} {:>14} {:>9} {:>9} {:>8} {:>8} {:>8} {:>8}", "shape", "m00", "centroid", "mu20/m00", "mu02/m00", "skew x", "kurt x", "skew y", "kurt y");
    let mut strip = Image::new(6 * 66, 64, 0.5f32);
    for (k, (name, img)) in shapes.iter().enumerate() {
        let m = raw_moments(img, 1);
        let mu = central_moments(img, 4);
        let (gx, kx, gy, ky) = skewness_kurtosis(&mu);
        let (xbar, ybar) = (m[1][0] / m[0][0], m[0][1] / m[0][0]);
        println!(
            "{name:>10} {:6.1} ({xbar:5.2}, {ybar:5.2}) {:9.2} {:9.2} {gx:8.3} {kx:8.3} {gy:8.3} {ky:8.3}",
            m[0][0],
            mu[2][0] / m[0][0],
            mu[0][2] / m[0][0]
        );
        for (x, y) in img.coords() {
            strip[(k * 66 + x, y)] = img[(x, y)];
        }
    }
    let big = Image::from_fn(strip.width() * 3, strip.height() * 3, |x, y| strip[(x / 3, y / 3)]);
    save_gray(&big, format!("{dir}/shapes.png"));

    // The micrograph up to order 10: directly and separably, in pixel units and in [-1, 1].
    let cell = samples::cell();
    let t = Instant::now();
    let direct = raw_moments(&cell, 10);
    let t_direct = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let sep = moments_separable(&cell, 10, false);
    let t_sep = t.elapsed().as_secs_f64();
    let scaled = moments_separable(&cell, 10, true);
    println!();
    println!("micrograph, all 66 moments up to order 10: direct {:.1} ms, separable {:.1} ms", 1e3 * t_direct, 1e3 * t_sep);
    println!("largest relative difference between the two: {:.1e}", rel_diff(&direct, &sep));
    println!("m00 = {:.4e} in both; m(10,0) = {:.4e} in pixel units, {:.4e} in [-1, 1]", direct[0][0], direct[10][0], scaled[10][0]);
    println!("wrote {dir}/shapes.png (pear, mirror, square, disc, L, grey pear)");
}

fn rel_diff(a: &Moments, b: &Moments) -> f64 {
    let mut worst = 0.0f64;
    for p in 0..a.len() {
        for q in 0..a.len() - p {
            worst = worst.max((a[p][q] - b[p][q]).abs() / a[p][q].abs().max(1e-300));
        }
    }
    worst
}
