// Prediction on the micrograph, on a smooth synthetic image with more and more noise, and on
// a flat field. Entropies are printed; residual pictures go to out/lossless/.
//   cargo run -p lossless                      (your code)
//   cargo run -p lossless --features solution  (the model solution)

use imgcourse::*;
use lossless::*;

/// Grey levels 0..=255 as whole numbers.
fn to_levels(img: &Image<f32>) -> Image<i32> {
    img.map(|v| (v * 255.0).round().clamp(0.0, 255.0) as i32)
}

/// A smooth synthetic 256 × 256 image: a few soft blobs on a gentle ramp, with noise of
/// standard deviation `sigma` grey levels.
fn smooth(sigma: f32, seed: u64) -> Image<i32> {
    let blobs = [(70.0, 80.0, 40.0, 90.0), (180.0, 100.0, 55.0, 70.0), (120.0, 190.0, 35.0, -60.0), (200.0, 210.0, 25.0, 50.0)];
    let clean = Image::from_fn(256, 256, |x, y| {
        let (x, y) = (x as f32, y as f32);
        let mut v = 60.0 + 0.25 * x + 0.15 * y;
        for (cx, cy, r, h) in blobs {
            v += h * (-((x - cx).powi(2) + (y - cy).powi(2)) / (2.0 * r * r)).exp();
        }
        v / 255.0
    });
    to_levels(&samples::add_noise(&clean, sigma / 255.0, seed))
}

fn table(name: &str, img: &Image<i32>) {
    print!("{name:<26}");
    for p in ALL {
        print!(" {:>7.2}", entropy(residuals(img, p).pixels()));
    }
    println!();
}

fn main() {
    let dir = "out/lossless";
    print!("{:<26}", "entropy (bits/pixel)");
    for p in ALL {
        print!(" {:>7}", format!("{p:?}"));
    }
    println!();

    let cell = to_levels(&samples::cell());
    table("cell micrograph", &cell);
    for sigma in [0.0, 1.5, 5.0, 10.0, 20.0] {
        table(&format!("smooth, noise sd {sigma}"), &smooth(sigma, 1));
    }
    let flat = to_levels(&samples::add_noise(&Image::new(256, 256, 128.0 / 255.0), 10.0 / 255.0, 2));
    table("flat 128, noise sd 10", &flat);

    // Pictures: the images and their residuals (mid grey = 0, contrast × 4: ±32 is white/black).
    let show = |img: &Image<i32>| img.map(|v| v as f32 / 255.0);
    let signed = |img: &Image<i32>| img.map(|v| 0.5 + v as f32 / 64.0);
    let sm = smooth(1.5, 1);
    save_gray(&show(&sm), format!("{dir}/smooth.png"));
    save_gray(&signed(&residuals(&sm, Predictor::Left)), format!("{dir}/smooth-residual-left.png"));
    save_gray(&show(&cell), format!("{dir}/cell.png"));
    save_gray(&signed(&residuals(&cell, Predictor::Left)), format!("{dir}/cell-residual-left.png"));
    save_gray(&signed(&residuals(&cell, Predictor::Med)), format!("{dir}/cell-residual-med.png"));

    let back = decode(&residuals(&cell, Predictor::Med), Predictor::Med);
    println!("decoded the micrograph from its MED residuals: identical = {}", back == cell);
    println!("wrote pictures to {dir}/");
}
