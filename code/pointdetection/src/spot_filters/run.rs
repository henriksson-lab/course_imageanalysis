// Finds spots in a synthetic image with an uneven background, with three filters, and writes
// the responses and the detections to out/spot-filters/.
//   cargo run -p pointdetection -- spot-filters                      (your code)
//   cargo run -p pointdetection --features solution -- spot-filters  (the model solution)

use imgcourse::*;
use crate::common::spot_scene;
use crate::spot_filters::sim::{draw, image_noise, score};
use crate::spot_filters::*;

pub fn main() {
    let dir = "out/spot-filters";
    // 80 spots of σ = 1.5 px, amplitudes 20–100, noise 10 (grey levels / 255), on a slope with a
    // broad bright patch at the upper right: the lecture's demo image.
    let g = |v: f32| v / 255.0;
    let background = |x: f32, y: f32| g(40.0 + 0.2 * x + 60.0 * (-((x - 180.0).powi(2) + (y - 70.0).powi(2)) / (2.0 * 30.0f32.powi(2))).exp());
    let (img, truth) = spot_scene(256, 80, 1.5, (g(20.0), g(100.0)), g(10.0), background, 1);
    save_gray_stretched(&img, format!("{dir}/input.png"));
    let sigma_n = image_noise(&img);
    println!("{} true spots; noise estimated from the image: {:.1} grey levels (true 10)", truth.len(), sigma_n * 255.0);

    let report = |name: &str, det: &[(usize, usize)]| {
        let (tp, fp, fnn) = score(det, &truth, 2.0);
        let f1 = 2.0 * tp as f32 / (2 * tp + fp + fnn) as f32;
        println!("{name:<28} {:>5} kept   TP {tp:>3}  FP {fp:>4}  FN {fnn:>3}  F1 {f1:.2}", det.len());
    };

    // Candidates without any filter: every noise wiggle is a local maximum.
    println!("local maxima without a filter (r = 1): {}", local_maxima(&img, 1).len());

    // Filter, local maxima (r = 2), threshold at k = 4 noise sd of the filtered image.
    for (name, zero_mean) in [("gaussian", false), ("zero-mean", true)] {
        let k = spot_kernel(1.5, zero_mean);
        let response = correlate(&img, &k);
        let noise_after = sigma_n * k.pixels().iter().map(|w| w * w).sum::<f32>().sqrt();
        let candidates = local_maxima(&response, 2);
        // Gaussian: the response includes the background, so measure from the median.
        let mut sorted = response.pixels().to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let base = if zero_mean { 0.0 } else { sorted[sorted.len() / 2] };
        let kept: Vec<_> = candidates.iter().copied().filter(|&(x, y)| response[(x, y)] - base > 4.0 * noise_after).collect();
        println!("{name}: {} candidates, noise after the filter {:.2} grey levels", candidates.len(), noise_after * 255.0);
        report(&format!("{name} filter, k = 4"), &kept);
        save_gray_signed(&response.map(|v| v - base), format!("{dir}/response-{name}.png"));
        save_rgb(&draw(&img, &kept, &truth, 2.0), format!("{dir}/detections-{name}.png"));
    }

    // À trous: levels w1..w4, product of w2 and w3 at k = 3 per level.
    let (w, last) = atrous(&img, 4);
    for (j, wj) in w.iter().enumerate() {
        save_gray_signed(wj, format!("{dir}/atrous-w{}.png", j + 1));
    }
    save_gray_stretched(&last, format!("{dir}/atrous-a4.png"));
    let p = multiscale_product(&w, &[2, 3], 3.0);
    let kept: Vec<_> = local_maxima(&p, 2).into_iter().filter(|&(x, y)| p[(x, y)] > 0.0).collect();
    report("a trous product w2·w3, k = 3", &kept);
    save_gray_stretched(&p.map(|v| v.sqrt()), format!("{dir}/atrous-product.png"));
    save_rgb(&draw(&img, &kept, &truth, 2.0), format!("{dir}/detections-atrous.png"));
    let p1 = multiscale_product(&w, &[1], 3.0);
    let kept1: Vec<_> = local_maxima(&p1, 2).into_iter().filter(|&(x, y)| p1[(x, y)] > 0.0).collect();
    report("a trous, only w1, k = 3", &kept1);
    println!("wrote the pictures to {dir}/ (green = found, red = false, yellow ring = missed)");
}
