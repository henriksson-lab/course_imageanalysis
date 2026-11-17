// Compares your fast marching with Dijkstra on a flat cost, and finds a minimal path with an
// image cost; results in out/fastmarching/.
//   cargo run -p dynprog -- fastmarching                      (your code)
//   cargo run -p dynprog --features solution -- fastmarching  (the model solution)

use crate::fastmarching::*;
use crate::common::smooth;
use imgcourse::*;

/// Arrival times as a picture: dark = early, with white fronts every `every` time units.
fn fronts(t: &Image<f32>, every: f32, t_max: f32) -> Image<[f32; 3]> {
    t.map_xy(|x, y, v| {
        let band = |x: usize, y: usize| (t[(x, y)] / every).floor();
        let right = x + 1 < t.width() && band(x + 1, y) != band(x, y);
        let below = y + 1 < t.height() && band(x, y + 1) != band(x, y);
        if right || below {
            [1.0, 1.0, 1.0]
        } else {
            let g = (v / t_max).min(1.0);
            [0.1 + 0.8 * g, 0.2 + 0.4 * g, 0.5 - 0.4 * g]
        }
    })
}

pub fn main() {
    let dir = "out/fastmarching";

    // 1. Flat cost, one seed in the middle: circles, diamonds and octagons.
    let n = 201;
    let flat = Image::new(n, n, 1.0);
    let seed = [(100, 100)];
    let start = std::time::Instant::now();
    let fm = fast_marching(&flat, &seed);
    let ms = start.elapsed().as_secs_f64() * 1e3;
    let d4 = dijkstra_time(&flat, &seed, false);
    let d8 = dijkstra_time(&flat, &seed, true);
    let mut sheet = Image::new(3 * n + 20, n, [1.0f32; 3]);
    for (i, t) in [&fm, &d4, &d8].iter().enumerate() {
        let pic = fronts(t, 10.0, 140.0);
        for (x, y) in pic.coords() {
            sheet[(i * (n + 10) + x, y)] = pic[(x, y)];
        }
    }
    save_rgb(&sheet, format!("{dir}/fronts-fm-d4-d8.png"));
    println!("flat cost, {n} × {n}: fast marching took {ms:.0} ms");
    for (name, (x, y)) in [("0°", (195, 100)), ("22.5°", (192, 138)), ("45°", (170, 170))] {
        let r = (((x - 100) * (x - 100) + (y - 100) * (y - 100)) as f32).sqrt();
        let p = |t: &Image<f32>| format!("{:.1} ({:+.1} %)", t[(x, y)], 100.0 * (t[(x, y)] / r - 1.0));
        println!("  at {name}, ({x}, {y}), Euclidean {r:.1}: fast marching {}, Dijkstra 4 {}, Dijkstra 8 {}", p(&fm), p(&d4), p(&d8));
    }

    // 2. An image cost, slow at edges: P = 1 + (|∇I| / k)², seed inside a nucleus.
    let (img, _) = samples::nuclei(512, 14, 7);
    let s = smooth(&img, 1.5);
    let k = 0.03;
    let cost = s.map_xy(|x, y, _| {
        let (x, y) = (x as isize, y as isize);
        let gx = (s.get_border(x + 1, y, Border::Clamp) - s.get_border(x - 1, y, Border::Clamp)) / 2.0;
        let gy = (s.get_border(x, y + 1, Border::Clamp) - s.get_border(x, y - 1, Border::Clamp)) / 2.0;
        1.0 + (gx * gx + gy * gy) / (k * k)
    });
    let (seed, target) = ((128, 143), (215.0f32, 250.0f32));
    let start = std::time::Instant::now();
    let t = fast_marching(&cost, &[seed]);
    let ms = start.elapsed().as_secs_f64() * 1e3;
    let t_flat = fast_marching(&Image::new(512, 512, 1.0), &[seed]);
    let path = descend(&t, target, 0.5);
    let target_px = (target.0 as usize, target.1 as usize);
    println!(
        "edge cost (k = {k}), 512 × 512: T at the target {:.1}, with P = 1: {:.1}; path of {:.1} px; {ms:.0} ms",
        t[target_px],
        t_flat[target_px],
        0.5 * (path.len() - 1) as f32
    );
    let mut pic = fronts(&t, 10.0, 250.0);
    for (x, y) in pic.coords() {
        if pic[(x, y)] != [1.0; 3] {
            let g = img[(x, y)];
            pic[(x, y)] = [0.5 * pic[(x, y)][0] + 0.5 * g, 0.5 * pic[(x, y)][1] + 0.5 * g, 0.5 * pic[(x, y)][2] + 0.5 * g];
        }
    }
    for &(x, y) in &path {
        pic[(x.round() as usize, y.round() as usize)] = [1.0, 0.5, 0.0];
    }
    save_rgb(&pic.crop(40, 50, 260, 260), format!("{dir}/edge-cost-fronts.png"));
    println!("wrote the results to {dir}/");
}
