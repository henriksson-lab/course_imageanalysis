// Finds the boundary of a synthetic nucleus with your polar DP; results in out/polar-boundary/.
//   cargo run -p dynprog -- polar-boundary                      (your code)
//   cargo run -p dynprog --features solution -- polar-boundary  (the model solution)

use imgcourse::*;
use crate::polar_boundary::*;
use crate::common::smooth;
use std::f32::consts::PI;

/// The closed polygon of a path (radius index at every angle), drawn on a colour image.
fn draw_contour(rgb: &mut Image<[f32; 3]>, cx: f32, cy: f32, radii: &[usize], colour: [f32; 3]) {
    let k_count = radii.len();
    let point = |k: usize| {
        let theta = 2.0 * PI * k as f32 / k_count as f32;
        (cx + radii[k % k_count] as f32 * theta.cos(), cy + radii[k % k_count] as f32 * theta.sin())
    };
    for k in 0..k_count {
        let ((x0, y0), (x1, y1)) = (point(k), point(k + 1));
        for i in 0..=20 {
            let t = i as f32 / 20.0;
            let (x, y) = ((x0 + t * (x1 - x0)).round() as isize, (y0 + t * (y1 - y0)).round() as isize);
            if rgb.contains(x, y) {
                rgb[(x as usize, y as usize)] = colour;
            }
        }
    }
}

/// Dice overlap between the region inside the contour and the true nucleus `label`.
fn dice(labels: &Image<u32>, label: u32, cx: f32, cy: f32, radii: &[usize]) -> f32 {
    let k_count = radii.len() as f32;
    let (mut both, mut inside, mut truth) = (0.0, 0.0, 0.0);
    for (x, y) in labels.coords() {
        let (dx, dy) = (x as f32 - cx, y as f32 - cy);
        let theta = dy.atan2(dx).rem_euclid(2.0 * PI);
        let k = (theta / (2.0 * PI) * k_count).round() as usize % radii.len();
        let a = (dx * dx + dy * dy).sqrt() <= radii[k] as f32;
        let b = labels[(x, y)] == label;
        both += (a && b) as u8 as f32;
        inside += a as u8 as f32;
        truth += b as u8 as f32;
    }
    2.0 * both / (inside + truth)
}

pub fn main() {
    let dir = "out/polar-boundary";
    let (img, labels) = samples::nuclei(512, 14, 7);
    save_gray(&img, format!("{dir}/input.png"));

    // The nucleus with the most pixels; its centre is the centroid of the true label.
    let mut area = [0usize; 15];
    for &l in labels.pixels() {
        area[l as usize] += 1;
    }
    let label = (1..area.len()).max_by_key(|&l| area[l]).unwrap() as u32;
    let (mut cx, mut cy) = (0.0, 0.0);
    for (x, y) in labels.coords() {
        if labels[(x, y)] == label {
            cx += x as f32 / area[label as usize] as f32;
            cy += y as f32 / area[label as usize] as f32;
        }
    }
    println!("nucleus {label}: {} pixels, centre ({cx:.1}, {cy:.1})", area[label as usize]);

    let (angles, radii) = (120, 90);
    let polar = unwrap(&smooth(&img, 1.5), cx, cy, angles, radii);
    let cost = radial_cost(&polar, Polarity::BrightToDark);
    save_gray(&polar, format!("{dir}/polar.png"));
    save_gray_signed(&cost, format!("{dir}/cost.png"));

    let mut contours = gray_to_rgb(&img);
    let mut unwrapped = gray_to_rgb(&polar);
    let runs: [(&str, usize, f32, bool, [f32; 3]); 3] = [
        ("open, delta 1, alpha 0.02", 1, 0.02, false, [0.2, 0.5, 1.0]),
        ("closed, delta 1, alpha 0.02", 1, 0.02, true, [1.0, 0.5, 0.0]),
        ("closed, delta 8, alpha 0", 8, 0.0, true, [1.0, 0.2, 0.8]),
    ];
    for (name, delta, alpha, closed, colour) in runs {
        let start = std::time::Instant::now();
        let (path, total) = if closed { closed_path(&cost, delta, alpha) } else { open_path(&cost, delta, alpha) };
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let gap = path[0].abs_diff(path[angles - 1]);
        let (lo, hi) = (path.iter().min().unwrap(), path.iter().max().unwrap());
        println!(
            "{name}: cost {total:.3}, gap {gap} px, radius {lo}..{hi}, Dice {:.3}, {ms:.1} ms",
            dice(&labels, label, cx, cy, &path)
        );
        if delta == 1 {
            draw_contour(&mut contours, cx, cy, &path, colour);
        } else {
            let mut rgb = gray_to_rgb(&img);
            draw_contour(&mut rgb, cx, cy, &path, colour);
            save_rgb(&rgb, format!("{dir}/contour-no-smoothness.png"));
        }
        if delta == 1 {
            for (k, &r) in path.iter().enumerate() {
                unwrapped[(k, r)] = colour;
            }
        }
    }
    save_rgb(&contours, format!("{dir}/contour.png"));
    save_rgb(&unwrapped, format!("{dir}/polar-paths.png"));
    println!("wrote the results to {dir}/");
}
