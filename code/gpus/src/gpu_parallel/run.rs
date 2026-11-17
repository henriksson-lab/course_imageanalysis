// A model GPU at work: branch divergence on synthetic nuclei for two warp shapes, a loop that
// waits for its slowest lane, and latency hiding. Numbers on the terminal, a picture of the
// warps in out/gpu-parallel/.
//   cargo run -p gpus -- gpu-parallel                      (your code)
//   cargo run -p gpus --features solution -- gpu-parallel  (the model solution)

use crate::gpu_parallel::*;
use imgcourse::*;

/// Colour every warp: orange = all lanes inside (path A only), blue = all outside (B only),
/// red = mixed (both paths).
fn warp_picture(mask: &Image<bool>, warp_w: usize, warp_h: usize) -> Image<[f32; 3]> {
    Image::from_fn(mask.width(), mask.height(), |x, y| {
        let (tx, ty) = (x / warp_w * warp_w, y / warp_h * warp_h);
        let lanes: Vec<bool> = (0..warp_w * warp_h).map(|k| mask[(tx + k % warp_w, ty + k / warp_w)]).collect();
        let base = match (lanes.iter().all(|&b| b), lanes.iter().all(|&b| !b)) {
            (true, _) => [0.95, 0.6, 0.2],
            (_, true) => [0.35, 0.55, 0.9],
            _ => [0.85, 0.15, 0.15],
        };
        // Darker where the pixel itself is inside, and a grid line at the warp borders.
        let border = (warp_w > 1 && x % warp_w == 0) || (warp_h > 1 && y % warp_h == 0);
        let f = if border { 0.4 } else if mask[(x, y)] { 0.75 } else { 1.0 };
        base.map(|c| c * f)
    })
}

pub fn main() {
    let dir = "out/gpu-parallel";
    let (img, labels) = samples::nuclei(256, 40, 3);
    save_gray(&img, format!("{dir}/nuclei.png"));
    let mask = labels.map(|l| l > 0);

    println!("Branch: 100 cycles inside a nucleus, 10 outside; 256 × 256 pixels, 2048 warps");
    let ideal: usize = mask.pixels().iter().map(|&a| if a { 100 } else { 10 }).sum::<usize>() / WARP;
    for (w, h) in [(32, 1), (8, 4), (4, 8)] {
        let d = divergence(&mask, w, h, 100, 10);
        println!(
            "  warps of {w:2} × {h}: {:4} of {} diverge, {} cycles (without divergence {ideal}), efficiency {:.1} %",
            d.diverged, d.warps, d.cycles, 100.0 * d.efficiency
        );
        save_rgb(&warp_picture(&mask, w, h), format!("{dir}/warps-{w}x{h}.png"));
    }

    // A loop whose length depends on the pixel: 1 iteration in the background, 2 to 20 in
    // the nuclei (brighter = more), as for "iterate until converged".
    let iters: Vec<usize> = img.pixels().iter().zip(mask.pixels()).map(|(&v, &m)| if m { 2 + (18.0 * v.clamp(0.0, 1.0)) as usize } else { 1 }).collect();
    let (mut warp_total, mut useful) = (0, 0);
    for chunk in iters.chunks(WARP) {
        warp_total += loop_efficiency(chunk).0;
        useful += chunk.iter().sum::<usize>();
    }
    println!(
        "Loop, warps = rows of 32: {} warp-iterations, {:.1} average per lane; lane efficiency {:.1} %",
        warp_total,
        useful as f32 / iters.len() as f32,
        100.0 * useful as f32 / (WARP * warp_total) as f32
    );

    println!("Latency hiding: k compute instructions per load of latency L = 200 cycles, 20 loads per warp");
    println!("  warps   k = 4: simulated (N k/(k+L))   k = 8: simulated (N k/(k+L))");
    for n in [1, 2, 4, 8, 16, 26, 32, 64] {
        let little = |k: f32| (n as f32 * k / (k + 200.0)).min(1.0);
        println!(
            "  {n:5}   {:5.1} % ({:5.1} %)              {:5.1} % ({:5.1} %)",
            100.0 * utilisation(n, 4, 200, 20),
            100.0 * little(4.0),
            100.0 * utilisation(n, 8, 200, 20),
            100.0 * little(8.0)
        );
    }
    println!("wrote the pictures to {dir}/");
}
