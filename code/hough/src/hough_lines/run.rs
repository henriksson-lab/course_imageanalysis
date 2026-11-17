// Finds straight lines in a synthetic scene of shapes with clutter: full voting against
// gradient voting, peaks with NMS, and segments. Pictures in out/hough-lines/.
//   cargo run -p hough -- hough-lines                      (your code)
//   cargo run -p hough --features solution -- hough-lines  (the model solution)

use crate::common::{draw_line, draw_segment, edge_points, polygon_scene, show_points};
use crate::hough_lines::*;
use imgcourse::*;

pub fn main() {
    let dir = "out/hough-lines";
    let size = 192;
    // A triangle, a quadrilateral, a thin bar, and 30 round clutter blobs.
    let mut shapes = vec![
        (vec![[20.0, 170.0], [95.0, 30.0], [120.0, 150.0]], 0.8),
        (vec![[110.0, 40.0], [175.0, 55.0], [165.0, 110.0], [105.0, 95.0]], 0.6),
        (vec![[15.0, 20.0], [180.0, 175.0], [176.0, 179.0], [11.0, 24.0]], 0.9),
    ];
    let mut rng = Rng::new(4);
    for _ in 0..30 {
        let (cx, cy, r) = (8.0 + 176.0 * rng.uniform(), 8.0 + 176.0 * rng.uniform(), 2.5 + 4.0 * rng.uniform());
        let blob: Vec<[f32; 2]> = (0..24).map(|i| {
            let a = i as f32 * std::f32::consts::TAU / 24.0;
            [cx + r * a.cos(), cy + r * a.sin()]
        }).collect();
        shapes.push((blob, 0.45));
    }
    let img = polygon_scene(size, 0.15, &shapes, 0.03, 1);
    save_gray(&img, format!("{dir}/scene.png"));
    let edges = edge_points(&img, 1.5, 0.25);
    let centre = [size as f32 / 2.0, size as f32 / 2.0];
    let pts: Vec<[f32; 2]> = edges.iter().map(|e| [e.x - centre[0], e.y - centre[1]]).collect();
    save_rgb(&show_points(&img, &edges), format!("{dir}/edges.png"));

    let (n_theta, d_rho) = (180, 1.0);
    let rho_max = ((size as f32 / 2.0) * 2f32.sqrt()).ceil();
    let full = accumulate(&pts, n_theta, d_rho, rho_max);
    let grad = accumulate_gradient(&pts, &edges, n_theta, d_rho, rho_max, 5);
    println!("{} edge points; full voting {} votes, gradient voting (±5°) {} votes", pts.len(), full.sum(), grad.sum());

    for (name, acc) in [("full", &full), ("gradient", &grad)] {
        save_gray_stretched(&acc.map(|v| v.sqrt()), format!("{dir}/accumulator-{name}.png"));
        let nms = peaks(acc, 30.0, 4);
        let no_nms = peaks(acc, 30.0, 0);
        println!("{name}: {} peaks above 30 votes with NMS ±4, {} without NMS; strongest:", nms.len(), no_nms.len());
        let mut out = show_points(&img, &edges);
        for &(j, k, v) in nms.iter().take(30) {
            let (theta, rho) = (theta_of(j, n_theta), rho_of(k, d_rho, rho_max));
            if v >= 30.0 {
                println!("   θ = {:>3}°, ρ = {:>6.1}: {v} votes", j, rho);
            }
            draw_line(&mut out, theta, rho, centre, [0.0, 0.9, 1.0]);
        }
        save_rgb(&out, format!("{dir}/lines-{name}.png"));
    }

    // Segments of the gradient-voting peaks: gap 4 px, at least 20 px long.
    let mut out = show_points(&img, &edges);
    let mut count = 0;
    for &(j, k, _) in &peaks(&grad, 30.0, 4) {
        for (a, b) in segments(&pts, theta_of(j, n_theta), rho_of(k, d_rho, rho_max), 1.0, 4.0, 20.0) {
            let shift = |p: [f32; 2]| [p[0] + centre[0], p[1] + centre[1]];
            draw_segment(&mut out, shift(a), shift(b), [0.0, 0.9, 1.0]);
            count += 1;
        }
    }
    println!("{count} segments (gap 4 px, at least 20 px)");
    save_rgb(&out, format!("{dir}/segments.png"));
    println!("wrote the pictures to {dir}/");
}
