// Brute-force registration of two crops of the micrograph, and template matching.
// Results in out/matching/.
//   cargo run -p day23 -- matching                      (your code)
//   cargo run -p day23 --features solution -- matching  (the model solution)

use imgcourse::*;
use crate::matching::*;
use std::time::Instant;

pub fn main() {
    let dir = "out/matching";
    let cell = samples::cell();

    // Two overlapping 128 × 128 tiles; B is 13 px right and 7 px down of A, and brighter.
    let a = cell.crop(40, 50, 128, 128);
    let b = cell.crop(53, 57, 128, 128).map(|v| 0.8 * v + 0.15);
    save_gray(&a, format!("{dir}/tile-a.png"));
    save_gray(&b, format!("{dir}/tile-b.png"));
    for w in [8, 20] {
        let t0 = Instant::now();
        let (tx, ty, c) = register(&a, &b, w, 1000);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let pairs: usize = (-w..=w).map(|t: isize| 128 - t.unsigned_abs()).sum::<usize>().pow(2);
        println!("search ±{w:2}: shift ({tx}, {ty}), NCC {c:.3}; {} shifts, {pairs} pixel pairs, {ms:.0} ms", (2 * w + 1).pow(2));
    }

    // The NCC of every shift within ±20 px as a picture (shift 0 in the middle).
    let surface = Image::from_fn(41, 41, |x, y| ncc_2d(&a, &b, x as isize - 20, y as isize - 20, 1000).unwrap_or(0.0));
    save_gray_signed(&surface, format!("{dir}/ncc-surface.png"));

    // Template matching: a 20 × 20 template on a dark granule of the micrograph.
    let (tx, ty) = (99, 59);
    let tpl = cell.crop(tx, ty, 20, 20);
    let t0 = Instant::now();
    let map = template_match(&cell, &tpl);
    println!("\ntemplate matching: {} positions in {:.0} ms", map.width() * map.height(), t0.elapsed().as_secs_f64() * 1000.0);
    save_gray_signed(&map, format!("{dir}/template-map.png"));

    // The best matches: local maxima, at least 10 px apart.
    let mut peaks: Vec<(f32, usize, usize)> = map.coords().map(|(x, y)| (map[(x, y)], x, y)).collect();
    peaks.sort_by(|p, q| q.0.total_cmp(&p.0));
    let mut kept: Vec<(f32, usize, usize)> = Vec::new();
    for p in peaks {
        if kept.iter().all(|k| k.1.abs_diff(p.1).max(k.2.abs_diff(p.2)) >= 10) {
            kept.push(p);
        }
        if kept.len() == 5 {
            break;
        }
    }
    let mut out = gray_to_rgb(&cell);
    for (i, &(c, x, y)) in kept.iter().enumerate() {
        println!("  match {}: ({x}, {y}), NCC {c:.2}", i + 1);
        let colour = if i == 0 { [1.0, 1.0, 0.0] } else { [0.0, 1.0, 1.0] };
        for d in 0..20 {
            for (px, py) in [(x + d, y), (x + d, y + 19), (x, y + d), (x + 19, y + d)] {
                out[(px, py)] = colour;
            }
        }
    }
    save_rgb(&out, format!("{dir}/template-matches.png"));
    println!("wrote the results to {dir}/");
}
