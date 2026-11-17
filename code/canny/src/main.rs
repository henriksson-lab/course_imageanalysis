// Runs Canny, step by step, on a synthetic scene and on the course micrograph; results in
// out/canny/.
//   cargo run -p canny                      (your code)
//   cargo run -p canny --features solution  (the model solution)

use canny::*;
use imgcourse::*;

/// A 128 × 128 test scene as in the lecture's demo: a bright rectangle, a disc, and on the
/// right a faint rectangle whose contrast fades from 45 (top) to 8 (bottom) grey levels,
/// on a background of 60, with noise of 12 grey levels. Values in [0, 1].
fn scene() -> Image<f32> {
    let clean = Image::from_fn(128, 128, |x, y| {
        let (xf, yf) = (x as f32, y as f32);
        let mut v = 60.0;
        if (14..50).contains(&x) && (14..50).contains(&y) {
            v = 170.0;
        }
        if (xf - 40.0).powi(2) + (yf - 90.0).powi(2) < 18.0f32.powi(2) {
            v = 120.0;
        }
        if (78..114).contains(&x) && (20..116).contains(&y) {
            v = 60.0 + 45.0 - 37.0 * (yf - 20.0) / 95.0;
        }
        v / 255.0
    });
    samples::add_noise(&clean, 12.0 / 255.0, 3)
}

/// Red where `edge` is true, on top of the grey image.
fn overlay(img: &Image<f32>, edge: &Image<bool>) -> Image<[f32; 3]> {
    img.zip_map(edge, |v, e| if e { [1.0, 0.25, 0.0] } else { [v, v, v] })
}

fn count(edge: &Image<bool>) -> usize {
    edge.pixels().iter().filter(|&&e| e).count()
}

fn main() {
    let dir = "out/canny";

    // The four steps on the scene, σ = 1.5. Thresholds in grey levels per pixel, as in the demo.
    let img = scene();
    save_gray(&img, format!("{dir}/scene.png"));
    let (lx, ly) = gradient(&img, 1.5);
    let mag = lx.zip_map(&ly, |a, b| (a * a + b * b).sqrt() * 255.0);
    save_gray_stretched(&mag, format!("{dir}/scene-magnitude.png"));
    let thin = non_max_suppression(&mag, &lx, &ly);
    save_gray_stretched(&thin, format!("{dir}/scene-nms.png"));
    let wide = mag.pixels().iter().filter(|&&m| m >= 3.0).count();
    let crest = thin.pixels().iter().filter(|&&m| m >= 3.0).count();
    println!("scene, σ = 1.5: {wide} pixels with magnitude ≥ 3 before NMS, {crest} after");

    for (name, low, high) in [("hysteresis-3-10", 3.0, 10.0), ("single-10", 10.0, 10.0), ("single-3", 3.0, 3.0)] {
        let edge = hysteresis(&thin, low, high);
        println!("  thresholds {low} and {high}: {} edge pixels", count(&edge));
        save_rgb(&overlay(&img, &edge), format!("{dir}/scene-{name}.png"));
    }
    for sigma in [0.7, 4.0] {
        let (lx, ly) = gradient(&img, sigma);
        let mag = lx.zip_map(&ly, |a, b| (a * a + b * b).sqrt() * 255.0);
        let edge = hysteresis(&non_max_suppression(&mag, &lx, &ly), 3.0, 10.0);
        println!("scene, σ = {sigma}, thresholds 3 and 10: {} edge pixels", count(&edge));
        save_rgb(&overlay(&img, &edge), format!("{dir}/scene-sigma{sigma}.png"));
    }

    // The micrograph at three scales; the thresholds are lowered as σ grows.
    let cell = samples::cell();
    for (sigma, low, high) in [(1.0, 8.0, 16.0), (2.0, 5.0, 10.0), (4.0, 2.5, 5.0)] {
        let (lx, ly) = gradient(&cell, sigma);
        let mag = lx.zip_map(&ly, |a, b| (a * a + b * b).sqrt() * 255.0);
        let edge = hysteresis(&non_max_suppression(&mag, &lx, &ly), low, high);
        println!("micrograph, σ = {sigma}, thresholds {low} and {high}: {} edge pixels", count(&edge));
        save_rgb(&overlay(&cell, &edge), format!("{dir}/cell-sigma{sigma}.png"));
    }
    println!("wrote the results to {dir}/");
}
