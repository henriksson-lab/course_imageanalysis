// Cross-correlation of two crops of the micrograph, with the FFT and directly: circular,
// zero-padded, and how long each takes. Pictures in out/xcorr-fft/ (shift 0 in the middle).
//   cargo run -p fftreg -- xcorr-fft                      (your code)
//   cargo run -p fftreg --features solution -- xcorr-fft  (the model solution)

use imgcourse::*;
use std::time::Instant;
use crate::common::centred;
use crate::xcorr_fft::*;

/// `img − its mean`, in the top-left corner of a `size × size` array of zeros.
fn pad(img: &Image<f32>, size: usize) -> Image<f32> {
    let m = img.mean();
    Image::from_fn(size, size, |x, y| if x < img.width() && y < img.height() { img[(x, y)] - m } else { 0.0 })
}

pub fn main() {
    let dir = "out/xcorr-fft";
    let cell = samples::cell();
    let n = 128;
    for (tx, ty) in [(21, 13), (70, 10)] {
        let a = cell.crop(30, 40, n, n);
        let b = cell.crop(30 + tx, 40 + ty, n, n);
        println!("B is ({tx}, {ty}) px right and down of A:");
        let circ = xcorr2(&a, &b);
        let (px, py, _) = peak(&circ);
        println!("  circular:          peak at ({px}, {py})");
        let lin = xcorr2(&pad(&a, 2 * n), &pad(&b, 2 * n));
        let (qx, qy, _) = peak(&lin);
        println!("  zero-padded to {}: peak at ({qx}, {qy})", 2 * n);
        save_gray_signed(&centred(&circ), format!("{dir}/circular-{tx}-{ty}.png"));
        save_gray_signed(&centred(&lin), format!("{dir}/padded-{tx}-{ty}.png"));
    }

    // The cost: every circular shift of two 64 × 64 crops, directly and with the FFT.
    let n = 64;
    let a = cell.crop(30, 40, n, n).map(|v| v - 0.5);
    let b = cell.crop(51, 53, n, n).map(|v| v - 0.5);
    let t0 = Instant::now();
    let direct = Image::from_fn(n, n, |tx, ty| {
        let mut s = 0.0;
        for (x, y) in b.coords() {
            s += a[((x + tx) % n, (y + ty) % n)] * b[(x, y)];
        }
        s
    });
    let t_direct = t0.elapsed().as_secs_f64() * 1000.0;
    let t0 = Instant::now();
    let (ma, mb) = (a.mean(), b.mean());
    let fast = xcorr2(&a, &b);
    let t_fft = t0.elapsed().as_secs_f64() * 1000.0;
    // xcorr2 subtracts the means; the direct sum did not: correct for it before comparing
    let diff = direct.zip_map(&fast, |d, f| (d - (n * n) as f32 * ma * mb - f).abs()).max();
    println!("\nall {} shifts of two {n} × {n} crops: directly {t_direct:.0} ms, with the FFT {t_fft:.1} ms (largest difference {diff:.1e})", n * n);
    println!("wrote the results to {dir}/");
}
