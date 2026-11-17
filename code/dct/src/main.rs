// The DCT on the micrograph: keep the largest coefficients of 8 × 8 blocks, or of one DCT of
// the whole image, and compare. Pictures go to out/dct/.
//   cargo run -p dct                      (your code)
//   cargo run -p dct --features solution  (the model solution)

use dct::*;
use imgcourse::*;

/// The share of the image's variation that `approx` keeps, and its RMS error.
fn quality(img: &Image<f32>, approx: &Image<f32>) -> (f32, f32) {
    let mean = img.mean();
    let variation: f32 = img.pixels().iter().map(|v| (v - mean).powi(2)).sum();
    let error: f32 = img.pixels().iter().zip(approx.pixels()).map(|(a, b)| (a - b).powi(2)).sum();
    (1.0 - error / variation, (error / img.pixels().len() as f32).sqrt())
}

/// Keep the largest `fraction` of the coefficients of one DCT of the whole image.
fn keep_largest_whole(img: &Image<f32>, fraction: f32) -> Image<f32> {
    let c = dct2(img);
    let mut sizes: Vec<f32> = c.pixels().iter().map(|v| v.abs()).collect();
    sizes.sort_by(|a, b| b.total_cmp(a));
    let threshold = sizes[(fraction * sizes.len() as f32).round() as usize - 1];
    idct2(&c.map(|v| if v.abs() >= threshold { v } else { 0.0 }))
}

fn main() {
    let dir = "out/dct";

    // The 64 basis images of the 8 × 8 DCT, side by side with a 1-pixel gap (0 = mid grey).
    let mut sheet = Image::new(8 * 9 - 1, 8 * 9 - 1, 1.0f32);
    for v in 0..8 {
        for u in 0..8 {
            let mut c = Image::new(8, 8, 0.0);
            c[(u, v)] = 1.0;
            let basis = idct2(&c);
            for (x, y) in basis.coords() {
                sheet[(u * 9 + x, v * 9 + y)] = 0.5 + 1.5 * basis[(x, y)];
            }
        }
    }
    let big = Image::from_fn(sheet.width() * 4, sheet.height() * 4, |x, y| sheet[(x / 4, y / 4)]);
    save_gray(&big, format!("{dir}/basis64.png"));

    let img = samples::cell();
    save_gray(&img, format!("{dir}/cell.png"));
    println!("{:>6}  {:>22}  {:>22}", "kept", "8 × 8 blocks", "whole image");
    for fraction in [0.01, 0.03, 0.10] {
        // Subtract mid grey first (JPEG's level shift), so that a block without any kept
        // coefficient comes out mid grey, not black.
        let shifted = img.map(|v| v - 0.5);
        let blocks = keep_largest_blocks(&shifted, fraction).map(|v| v + 0.5);
        let whole = keep_largest_whole(&shifted, fraction).map(|v| v + 0.5);
        let (vb, eb) = quality(&img, &blocks);
        let (vw, ew) = quality(&img, &whole);
        println!(
            "{:>5.0}%  {:>6.1}% kept, RMS {:.3}  {:>6.1}% kept, RMS {:.3}",
            100.0 * fraction,
            100.0 * vb,
            eb,
            100.0 * vw,
            ew
        );
        let pct = (100.0 * fraction).round();
        save_gray(&blocks, format!("{dir}/blocks-keep{pct}.png"));
        save_gray(&whole, format!("{dir}/whole-keep{pct}.png"));
    }
    println!("wrote pictures to {dir}/");
}
