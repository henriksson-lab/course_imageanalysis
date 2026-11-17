// Runs your histogram and threshold on synthetic blobs and on the micrograph; results in
// out/histogram/.
//   cargo run -p day12 -- histogram                      (your code)
//   cargo run -p day12 --features solution -- histogram  (the model solution)

use crate::histogram::*;
use imgcourse::*;

/// Bright discs (radius 9, every 32 pixels) on a dark background, about 25 % object.
fn blob_truth() -> Image<bool> {
    Image::from_fn(256, 256, |x, y| {
        let (dx, dy) = ((x % 32) as f32 - 15.5, (y % 32) as f32 - 15.5);
        dx * dx + dy * dy <= 81.0
    })
}

/// Grey levels `background` and `object` (0–255) on the true mask, plus noise of `sigma` grey levels.
fn blob_image(truth: &Image<bool>, background: f32, object: f32, sigma: f32, seed: u64) -> Image<f32> {
    let clean = truth.map(|t| if t { object / 255.0 } else { background / 255.0 });
    samples::add_noise(&clean, sigma / 255.0, seed)
}

/// The 3 × 3 mean (a box filter, as in the kernels unit), edge pixels copied outwards.
fn mean3x3(img: &Image<f32>) -> Image<f32> {
    img.map_xy(|x, y, _| {
        let mut sum = 0.0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                sum += img.get_border(x as isize + dx, y as isize + dy, Border::Clamp);
            }
        }
        sum / 9.0
    })
}

/// A picture of a histogram: one bar per bin (2 pixels wide), the threshold as a grey line.
fn histogram_picture(h: &[usize], t: usize) -> Image<f32> {
    let height = 150;
    let top = *h.iter().max().unwrap() as f32;
    Image::from_fn(2 * h.len(), height, |x, y| {
        let bar = (h[x / 2] as f32 / top * (height - 1) as f32).round() as usize;
        if height - 1 - y < bar {
            0.0
        } else if x / 2 == t {
            0.6
        } else {
            1.0
        }
    })
}

fn to_gray(mask: &Image<bool>) -> Image<f32> {
    mask.map(|b| if b { 1.0 } else { 0.0 })
}

/// The number of pixels where the mask and the truth disagree.
fn wrong(mask: &Image<bool>, truth: &Image<bool>) -> usize {
    mask.zip_map(truth, |m, t| m != t).pixels().iter().filter(|&&w| w).count()
}

pub fn main() {
    let dir = "out/histogram";
    let truth = blob_truth();
    let n = truth.pixels().len();
    let objects = truth.pixels().iter().filter(|&&t| t).count();
    println!("blobs: {objects} of {n} pixels are object ({:.1} %)", 100.0 * objects as f32 / n as f32);

    // A clean bimodal image: background 60, objects 160, noise 10.
    let img = blob_image(&truth, 60.0, 160.0, 10.0, 1);
    let h = histogram(&img, 256);
    let (mean, var) = mean_variance(&h);
    println!("clean blobs: mean grey level {mean:.1}, standard deviation {:.1}", var.sqrt());
    let p = cumulative(&h);
    for t in [100, 110, 120, 130] {
        println!("  t = {t}: mask area {:.2} %, wrong pixels {}", 100.0 * (1.0 - p[t]), wrong(&threshold(&img, 256, t), &truth));
    }
    let tp = percentile_threshold(&h, objects as f32 / n as f32);
    println!("  percentile threshold for the true object fraction: t = {tp}");
    save_gray(&img, format!("{dir}/blobs.png"));
    save_gray(&histogram_picture(&h, 110), format!("{dir}/blobs-histogram.png"));
    save_gray(&to_gray(&threshold(&img, 256, 110)), format!("{dir}/blobs-mask.png"));

    // B2: background 50, objects 110, noise 20, threshold 80; then a 3 × 3 mean first.
    let noisy = blob_image(&truth, 50.0, 110.0, 20.0, 2);
    let smooth = mean3x3(&noisy);
    let mask_noisy = threshold(&noisy, 256, 80);
    let mask_smooth = threshold(&smooth, 256, 80);
    println!("noise 20, t = 80: {} wrong pixels; after a 3 × 3 mean: {}", wrong(&mask_noisy, &truth), wrong(&mask_smooth, &truth));
    save_gray(&noisy, format!("{dir}/noisy.png"));
    save_gray(&histogram_picture(&histogram(&noisy, 256), 80), format!("{dir}/noisy-histogram.png"));
    save_gray(&histogram_picture(&histogram(&smooth, 256), 80), format!("{dir}/smooth-histogram.png"));
    save_gray(&to_gray(&mask_noisy), format!("{dir}/noisy-mask.png"));
    save_gray(&to_gray(&mask_smooth), format!("{dir}/smooth-mask.png"));
    // Where are the wrong pixels after smoothing? White = wrong.
    save_gray(&mask_smooth.zip_map(&truth, |m, t| if m != t { 1.0 } else { 0.0 }), format!("{dir}/smooth-wrong.png"));

    // The micrograph: its histogram, and the dark half of the pixels.
    let cell = samples::cell();
    let hc = histogram(&cell, 256);
    let t_half = percentile_threshold(&hc, 0.5);
    println!("micrograph: half of the pixels are above t = {t_half}");
    save_gray(&histogram_picture(&hc, t_half), format!("{dir}/cell-histogram.png"));
    println!("wrote the results to {dir}/");
}
