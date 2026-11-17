// Runs your grey-level morphology on bright spots and dark specks on an uneven background,
// and on the micrograph; results in out/morph-grey/.
//   cargo run -p morph-grey                      (your code)
//   cargo run -p morph-grey --features solution  (the model solution)

use imgcourse::*;
use morph_grey::*;

/// 40 bright spots (Gaussian, σ = 1.6) and 25 dark specks on a sloping, vignetted
/// background with noise. Returns the image and the centres of the bright spots.
fn spots() -> (Image<f32>, Vec<(f32, f32)>) {
    let mut rng = Rng::new(9);
    let mut centres: Vec<(f32, f32)> = Vec::new();
    while centres.len() < 65 {
        let c = (8.0 + 240.0 * rng.uniform(), 8.0 + 240.0 * rng.uniform());
        if centres.iter().all(|&(x, y)| (x - c.0).hypot(y - c.1) > 14.0) {
            centres.push(c);
        }
    }
    let heights: Vec<f32> = (0..65).map(|_| 0.15 + 0.25 * rng.uniform()).collect();
    let img = Image::from_fn(256, 256, |x, y| {
        let (xf, yf) = (x as f32, y as f32);
        let slope = 0.1 + 0.6 * (xf + 0.5 * yf) / 384.0;
        let rim = 1.0 - 0.3 * ((xf - 128.0).powi(2) + (yf - 128.0).powi(2)) / (2.0 * 128.0f32.powi(2));
        let mut v = slope * rim + 0.08;
        for (k, &(cx, cy)) in centres.iter().enumerate() {
            let g = (-((xf - cx).powi(2) + (yf - cy).powi(2)) / (2.0 * 1.6 * 1.6)).exp();
            // the first 40 are bright spots, the other 25 dark specks
            v += if k < 40 { heights[k] * g } else { -0.6 * heights[k] * g };
        }
        v
    });
    centres.truncate(40);
    (samples::add_noise(&img, 0.012, 10), centres)
}

/// How many spots have their centre above `t`, and how many pixels above `t` are farther
/// than 4 pixels from every spot centre (false pixels).
fn score(img: &Image<f32>, t: f32, centres: &[(f32, f32)]) -> (usize, usize) {
    let found = centres.iter().filter(|&&(x, y)| img[(x.round() as usize, y.round() as usize)] > t).count();
    let false_pixels = img
        .coords()
        .filter(|&(x, y)| img[(x, y)] > t && centres.iter().all(|&(cx, cy)| (x as f32 - cx).hypot(y as f32 - cy) > 4.0))
        .count();
    (found, false_pixels)
}

fn to_gray(img: &Image<f32>, t: f32) -> Image<f32> {
    img.map(|v| if v > t { 1.0 } else { 0.0 })
}

fn main() {
    let dir = "out/morph-grey";
    let (img, centres) = spots();
    save_gray(&img, format!("{dir}/spots.png"));
    for t in [0.40, 0.50, 0.60] {
        let (found, fp) = score(&img, t, &centres);
        println!("threshold the input at {t:.2}: {found} of 40 spots, {fp} false pixels");
    }
    save_gray(&to_gray(&img, 0.5), format!("{dir}/spots-threshold.png"));

    for r in [2, 6, 20] {
        let th = white_tophat(&img, &disc(r));
        let (found, fp) = score(&th, 0.1, &centres);
        println!("white top-hat, disc of radius {r:2}, threshold 0.10: {found} of 40 spots, {fp} false pixels");
        save_gray_stretched(&th, format!("{dir}/tophat-r{r}.png"));
        save_gray(&to_gray(&th, 0.1), format!("{dir}/tophat-r{r}-threshold.png"));
    }
    save_gray(&open(&img, &disc(6)), format!("{dir}/opening-r6.png"));
    save_gray_stretched(&black_tophat(&img, &disc(3)), format!("{dir}/black-tophat-r3.png"));

    // The micrograph: gradient, and salt-and-pepper noise against min, median and max.
    let cell = samples::cell();
    save_gray_stretched(&gradient(&cell, &disc(1)), format!("{dir}/cell-gradient-r1.png"));
    save_gray_stretched(&gradient(&cell, &disc(3)), format!("{dir}/cell-gradient-r3.png"));
    let mut rng = Rng::new(12);
    let salted = Image::from_fn(256, 256, |x, y| match rng.below(20) {
        0 => 0.0,
        1 => 1.0,
        _ => cell[(x, y)],
    });
    save_gray(&salted, format!("{dir}/cell-salt-pepper.png"));
    save_gray(&erode(&salted, &disc(1)), format!("{dir}/cell-min.png"));
    save_gray(&median(&salted, &disc(1)), format!("{dir}/cell-median.png"));
    save_gray(&dilate(&salted, &disc(1)), format!("{dir}/cell-max.png"));
    println!("wrote the results to {dir}/");
}
