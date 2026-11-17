// Runs your local thresholds on dark discs under uneven light, and hysteresis on faint
// filaments; results in out/local-threshold/.
//   cargo run -p thresholding -- local-threshold                      (your code)
//   cargo run -p thresholding --features solution -- local-threshold  (the model solution)

use imgcourse::*;
use crate::local_threshold::*;
use crate::common::to_gray;

/// Dark discs of radius 4 to 22 on a bright background, lit by light that is dim on the
/// left and bright on the right, with a darker rim (vignetting) and noise. Returns the image
/// and the true mask.
fn discs_under_uneven_light() -> (Image<f32>, Image<bool>) {
    let mut rng = Rng::new(7);
    let mut discs: Vec<(f32, f32, f32)> = Vec::new();
    while discs.len() < 30 {
        let r = 4.0 + 18.0 * rng.uniform() * rng.uniform();
        let (cx, cy) = (r + (255.0 - 2.0 * r) * rng.uniform(), r + (255.0 - 2.0 * r) * rng.uniform());
        if discs.iter().all(|&(x, y, q)| (x - cx).hypot(y - cy) > r + q + 3.0) {
            discs.push((cx, cy, r));
        }
    }
    let truth = Image::from_fn(256, 256, |x, y| discs.iter().any(|&(cx, cy, r)| (x as f32 - cx).hypot(y as f32 - cy) <= r));
    let mut img = Image::new(256, 256, 0.0);
    for (x, y) in img.coords() {
        let light = 0.31 + 0.48 * x as f32 / 255.0; // background about 79 on the left, 202 on the right
        let rim = 1.0 - 0.25 * ((x as f32 - 128.0).powi(2) + (y as f32 - 128.0).powi(2)) / (2.0 * 128.0f32.powi(2));
        let reflect = if truth[(x, y)] { 0.5 } else { 1.0 };
        img[(x, y)] = light * rim * reflect + 0.015 * rng.normal();
    }
    (img, truth)
}

/// Bright filaments that fade from one end to the other, plus noise.
fn fading_filaments() -> Image<f32> {
    let mut img = Image::new(128, 128, 0.0);
    for k in 0..4 {
        let y0 = 16.0 + 30.0 * k as f32;
        for xs in 0..1200 {
            let x = 8.0 + 112.0 * xs as f32 / 1200.0;
            let y = y0 + 8.0 * (x / 15.0 + k as f32).sin();
            let bright = 0.9 - 0.65 * (x - 8.0) / 112.0; // fades from 0.9 to 0.25
            img[(x as usize, y as usize)] = bright;
        }
    }
    samples::add_noise(&img.map(|v| v + 0.1), 0.07, 11)
}

/// (false positives, false negatives) of a mask against the truth.
fn errors(mask: &Image<bool>, truth: &Image<bool>) -> (usize, usize) {
    let fp = mask.zip_map(truth, |m, t| m && !t).pixels().iter().filter(|&&e| e).count();
    let fn_ = mask.zip_map(truth, |m, t| !m && t).pixels().iter().filter(|&&e| e).count();
    (fp, fn_)
}

pub fn main() {
    let dir = "out/local-threshold";
    let (img, truth) = discs_under_uneven_light();
    save_gray(&img, format!("{dir}/discs.png"));
    save_gray(&to_gray(&truth), format!("{dir}/discs-truth.png"));

    // The best global threshold (chosen with the truth, so no method can do better).
    let mut best = (0.0, usize::MAX);
    for k in 0..=255 {
        let t = k as f32 / 255.0;
        let (fp, fn_) = errors(&img.map(|v| v < t), &truth);
        if fp + fn_ < best.1 {
            best = (t, fp + fn_);
        }
    }
    let global = img.map(|v| v < best.0);
    let (fp, fn_) = errors(&global, &truth);
    println!("best global threshold {:.0}: {fp} false positives, {fn_} false negatives", 255.0 * best.0);
    save_gray(&to_gray(&global), format!("{dir}/global.png"));

    // Divide by the background (a 61 × 61 local mean), then one threshold: 0.85.
    let (background, _) = local_mean_std(&img, 30);
    let ratio = img.zip_map(&background, |v, b| v / b);
    let divided = ratio.map(|v| v < 0.85);
    let (fp, fn_) = errors(&divided, &truth);
    println!("divide by the background, ratio < 0.85: {fp} FP, {fn_} FN");
    save_gray_stretched(&ratio, format!("{dir}/ratio.png"));

    // mean − C (C = 10 grey levels) with growing windows.
    for r in [5, 25, 40, 120] {
        let mask = local_threshold(&img, r, Rule::MeanC(10.0 / 255.0));
        let (fp, fn_) = errors(&mask, &truth);
        println!("mean − C, window {0} × {0}: {fp} FP, {fn_} FN", 2 * r + 1);
        save_gray(&to_gray(&mask), format!("{dir}/mean-c-r{r}.png"));
    }
    for (name, rule) in [("niblack", Rule::Niblack(-0.2)), ("sauvola", Rule::Sauvola { k: 0.3, range: 0.5 })] {
        let mask = local_threshold(&img, 25, rule);
        let (fp, fn_) = errors(&mask, &truth);
        println!("{name}, window 51 × 51: {fp} FP, {fn_} FN");
        save_gray(&to_gray(&mask), format!("{dir}/{name}.png"));
    }

    // Hysteresis on fading filaments.
    let fil = fading_filaments();
    save_gray(&fil, format!("{dir}/filaments.png"));
    let (low, high) = (0.30, 0.55);
    let count = |m: &Image<bool>| m.pixels().iter().filter(|&&b| b).count();
    let single_high = fil.map(|v| v > high);
    let single_low = fil.map(|v| v > low);
    let hyst8 = hysteresis(&fil, low, high, true);
    let hyst4 = hysteresis(&fil, low, high, false);
    println!(
        "filaments: above high {}, above low {}, hysteresis {} (8-connected), {} (4-connected)",
        count(&single_high),
        count(&single_low),
        count(&hyst8),
        count(&hyst4)
    );
    save_gray(&to_gray(&single_high), format!("{dir}/filaments-high.png"));
    save_gray(&to_gray(&single_low), format!("{dir}/filaments-low.png"));
    save_gray(&to_gray(&hyst8), format!("{dir}/filaments-hysteresis.png"));
    println!("wrote the results to {dir}/");
}
