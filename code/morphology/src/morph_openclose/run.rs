// Runs your opening, closing, granulometry and distance transform on synthetic images;
// results in out/morph-openclose/.
//   cargo run -p morphology -- morph-openclose                      (your code)
//   cargo run -p morphology --features solution -- morph-openclose  (the model solution)

use imgcourse::*;
use crate::morph_openclose::*;
use crate::common::{count, to_f32};

/// Discs of radius 6 and 13 that do not touch, and a few specks of one or two pixels.
fn two_sizes() -> Image<bool> {
    let mut rng = Rng::new(5);
    let mut discs: Vec<(f32, f32, f32)> = Vec::new();
    for (radius, n) in [(13.0, 7), (6.0, 14)] {
        let mut placed = 0;
        while placed < n {
            let (cx, cy) = (radius + 1.0 + (254.0 - 2.0 * radius) * rng.uniform(), radius + 1.0 + (254.0 - 2.0 * radius) * rng.uniform());
            if discs.iter().all(|&(x, y, r)| (x - cx).hypot(y - cy) > r + radius + 4.0) {
                discs.push((cx, cy, radius));
                placed += 1;
            }
        }
    }
    let mut a = Image::from_fn(256, 256, |x, y| discs.iter().any(|&(cx, cy, r)| (x as f32 - cx).hypot(y as f32 - cy) <= r));
    for _ in 0..30 {
        let (x, y) = (rng.below(255), rng.below(256));
        a[(x, y)] = true;
        if rng.uniform() < 0.5 {
            a[(x + 1, y)] = true;
        }
    }
    a
}

/// A bar chart of the pattern spectrum, one bar of 8 pixels per radius.
fn bars(values: &[isize]) -> Image<f32> {
    let top = *values.iter().max().unwrap() as f32;
    let height = 120;
    Image::from_fn(8 * values.len(), height, |x, y| {
        let bar = (values[x / 8].max(0) as f32 / top * (height - 1) as f32).round() as usize;
        if height - 1 - y < bar && x % 8 != 7 {
            0.0
        } else {
            1.0
        }
    })
}

pub fn main() {
    let dir = "out/morph-openclose";

    // Granulometry with discs of radius 0 to 15.
    let a = two_sizes();
    save_gray(&to_f32(&a), format!("{dir}/two-sizes.png"));
    let ses: Vec<_> = (0..=15).map(disc).collect();
    let (area, spectrum) = granulometry(&a, &ses);
    println!("two sizes: {} object pixels", count(&a));
    for r in 0..spectrum.len() {
        println!("  r = {r:2}: area left {:5}, removed between r and r + 1: {:5}", area[r], spectrum[r]);
    }
    save_gray(&bars(&spectrum), format!("{dir}/pattern-spectrum.png"));

    // Noise removal on a thresholded image of noisy nuclei.
    let (img, _) = samples::nuclei(256, 25, 4);
    let noisy = samples::add_noise(&img, 0.12, 6).map(|v| v > 0.45);
    save_gray(&to_f32(&noisy), format!("{dir}/noisy-mask.png"));
    let se = disc(2);
    let results = [
        ("open", open(&noisy, &se)),
        ("close", close(&noisy, &se)),
        ("open-close", close(&open(&noisy, &se), &se)),
        ("close-open", open(&close(&noisy, &se), &se)),
    ];
    println!("noisy mask: {} pixels", count(&noisy));
    for (name, r) in &results {
        println!("  {name}: {} pixels", count(r));
        save_gray(&to_f32(r), format!("{dir}/{name}.png"));
    }

    // Distance transforms of the clean mask.
    let clean = img.map(|v| v > 0.45);
    for (name, chess) in [("city-block", false), ("chessboard", true)] {
        let d = distance_transform(&clean, chess);
        let largest = d.pixels().iter().max().unwrap();
        println!("{name} distance transform: largest value {largest}");
        save_gray_stretched(&d.map(|v| v as f32), format!("{dir}/dt-{name}.png"));
    }
    let r = 5;
    let same = erode_by_distance(&clean, r, false) == erode(&clean, &diamond(r as usize));
    println!("DT > {r} is the erosion by the diamond of radius {r}: {same}");
    println!("wrote the results to {dir}/");
}
