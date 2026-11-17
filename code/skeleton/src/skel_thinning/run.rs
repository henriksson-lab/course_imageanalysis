// Thins a few shapes (a rectangle, a ring, a synthetic neuron mask, the letter F) by
// templates and by simple points; counts and pictures in out/skel-thinning/.
//   cargo run -p skeleton -- skel-thinning                      (your code)
//   cargo run -p skeleton --features solution -- skel-thinning  (the model solution)

use imgcourse::*;
use crate::common::count;
use crate::skel_thinning::*;

/// A synthetic neuron mask: a round soma and five dendrites that branch, drawn as thick
/// lines that get thinner outwards, on a 200 × 200 image.
fn neuron() -> Image<bool> {
    // segments (x0, y0, x1, y1, half-width)
    let segs = [
        (100.0, 100.0, 40.0, 40.0, 5.0),
        (40.0, 40.0, 20.0, 12.0, 3.0),
        (40.0, 40.0, 10.0, 50.0, 3.0),
        (100.0, 100.0, 170.0, 60.0, 5.0),
        (170.0, 60.0, 185.0, 20.0, 3.0),
        (170.0, 60.0, 192.0, 80.0, 2.5),
        (100.0, 100.0, 120.0, 180.0, 4.0),
        (120.0, 180.0, 90.0, 194.0, 2.5),
        (100.0, 100.0, 30.0, 150.0, 4.0),
        (100.0, 100.0, 175.0, 150.0, 3.5),
        (140.0, 125.0, 150.0, 175.0, 2.5),
    ];
    Image::from_fn(200, 200, |x, y| {
        let (px, py) = (x as f32, y as f32);
        let soma = (px - 100.0).powi(2) + (py - 100.0).powi(2) < 14.0 * 14.0;
        soma || segs.iter().any(|&(x0, y0, x1, y1, w)| {
            // distance from the pixel to the segment
            let (dx, dy) = (x1 - x0, y1 - y0);
            let t = (((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
            (px - x0 - t * dx).powi(2) + (py - y0 - t * dy).powi(2) <= w * w
        })
    })
}

/// Mask in grey, skeleton in white, end points in blue, junctions (3 or more skeleton
/// neighbours) in red; enlarged `k` times.
fn picture(mask: &Image<bool>, skel: &Image<bool>, k: usize) -> Image<[f32; 3]> {
    let ends = hit_or_miss_all_turns(skel, &END_POINT);
    Image::from_fn(mask.width() * k, mask.height() * k, |x, y| {
        let (x, y) = (x / k, y / k);
        if ends[(x, y)] {
            return [0.2, 0.5, 1.0];
        }
        if skel[(x, y)] {
            let mut n = 0;
            for dy in -1..=1isize {
                for dx in -1..=1isize {
                    n += ((dx, dy) != (0, 0) && skel.get_border(x as isize + dx, y as isize + dy, Border::Zero)) as usize;
                }
            }
            return if n >= 3 { [1.0, 0.2, 0.2] } else { [1.0, 1.0, 1.0] };
        }
        if mask[(x, y)] { [0.35, 0.35, 0.35] } else { [0.0, 0.0, 0.0] }
    })
}

pub fn main() {
    let dir = "out/skel-thinning";
    let f = samples::letter_f(64).map(|v| v > 0.5);
    let ring = Image::from_fn(40, 40, |x, y| {
        let r = ((x as f32 - 19.5).powi(2) + (y as f32 - 19.5).powi(2)).sqrt();
        (7.0..16.0).contains(&r)
    });
    let rect = Image::from_fn(40, 20, |x, y| (4..36).contains(&x) && (4..16).contains(&y));
    let shapes = [("rectangle", rect, 8), ("ring", ring, 8), ("letter-f", f, 5), ("neuron", neuron(), 2)];

    println!("{:<10} {:>6} | {:>9} {:>6} {:>5} | {:>9} {:>5} | {:>9}", "shape", "pixels", "templates", "passes", "ends", "simple pt", "ends", "no ends");
    for (name, mask, k) in shapes {
        let start = std::time::Instant::now();
        let (skel, passes) = thin(&mask, false);
        let simple = thin_simple(&mask, true);
        let shrunk = thin_simple(&mask, false);
        let ends = |s: &Image<bool>| count(&hit_or_miss_all_turns(s, &END_POINT));
        println!(
            "{:<10} {:>6} | {:>9} {:>6} {:>5} | {:>9} {:>5} | {:>9}   ({:.2} s)",
            name,
            count(&mask),
            count(&skel),
            passes,
            ends(&skel),
            count(&simple),
            ends(&simple),
            count(&shrunk),
            start.elapsed().as_secs_f32()
        );
        save_rgb(&picture(&mask, &skel, k), format!("{dir}/{name}-templates.png"));
        save_rgb(&picture(&mask, &simple, k), format!("{dir}/{name}-simple.png"));
        save_rgb(&picture(&mask, &shrunk, k), format!("{dir}/{name}-no-end-points.png"));
    }

    // Parallel removal on the neuron: how much is left?
    let (par, _) = thin(&neuron(), true);
    println!(
        "neuron thinned in parallel (all 8 templates at once): {} pixels left, {} end points, {} isolated pixels",
        count(&par),
        count(&hit_or_miss_all_turns(&par, &END_POINT)),
        count(&hit_or_miss(&par, &ISOLATED))
    );
    save_rgb(&picture(&neuron(), &par, 2), format!("{dir}/neuron-parallel.png"));
    println!("wrote the pictures to {dir}/ (grey: mask, white: skeleton, blue: end points, red: junction pixels)");
}
