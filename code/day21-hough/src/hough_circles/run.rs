// Finds discs with the circle Hough transform: full voting against gradient voting on a scene
// of discs, an ellipse and an occluding band, then on synthetic nuclei. Pictures in out/hough-circles/.
//   cargo run -p day21 -- hough-circles                      (your code)
//   cargo run -p day21 --features solution -- hough-circles  (the model solution)

use crate::hough_circles::sim::{draw_circle, edge_points, ellipse, polygon_scene, show_points};
use crate::hough_circles::*;
use imgcourse::*;

/// All slices for the radii, with full voting (`None`) or gradient voting.
fn vote_all(edges: &[sim::EdgePoint], w: usize, h: usize, radii: &[f32], gradient: Option<Polarity>) -> Vec<Image<f32>> {
    radii
        .iter()
        .map(|&r| match gradient {
            None => ring_votes(edges, w, h, r),
            Some(p) => gradient_votes(edges, w, h, r, p),
        })
        .collect()
}

pub fn main() {
    let dir = "out/hough-circles";
    let size = 160;
    // Seven bright discs (one partly under a dark band), an ellipse, on a dark background.
    let discs = [(35.0, 35.0, 12.0), (53.0, 96.0, 12.0), (28.0, 135.0, 13.0), (140.0, 30.0, 8.0), (96.0, 32.0, 18.0), (62.0, 140.0, 7.0), (118.0, 128.0, 20.0)];
    let mut shapes: Vec<(Vec<[f32; 2]>, f32)> = discs.iter().map(|&(x, y, r)| (ellipse(x, y, r, r, 0.0), 0.8)).collect();
    shapes.push((ellipse(108.0, 82.0, 17.0, 10.0, 0.4), 0.8));
    shapes.push((vec![[100.0, 150.0], [150.0, 100.0], [156.0, 106.0], [106.0, 156.0]], 0.1));
    let img = polygon_scene(size, 0.15, &shapes, 0.03, 2);
    save_gray(&img, format!("{dir}/scene.png"));
    let edges = edge_points(&img, 1.0, 0.25);
    let radii: Vec<f32> = (5..=24).map(|r| r as f32).collect();
    println!("{} edge points, radii 5 to 24", edges.len());

    for (name, mode, threshold) in [("full", None, 0.6), ("gradient", Some(Polarity::BrightOnDark), 0.3), ("gradient-dark", Some(Polarity::DarkOnBright), 0.3)] {
        let slices = vote_all(&edges, size, size, &radii, mode);
        let votes: f32 = slices.iter().map(|s| s.sum()).sum();
        let found = find_circles(&slices, &radii, threshold);
        println!("{name}: {votes} votes, {} circles above {threshold}:", found.len());
        for c in &found {
            println!("   ({:>3}, {:>3}) r = {:>2}: {:.2}", c.x, c.y, c.r, c.score);
        }
        // the maximum over all radii, normalised by 2πr
        let best = Image::from_fn(size, size, |x, y| {
            slices.iter().zip(&radii).map(|(s, &r)| s[(x, y)] / (std::f32::consts::TAU * r)).fold(0.0, f32::max)
        });
        save_gray_stretched(&best, format!("{dir}/max-over-r-{name}.png"));
        let mut out = show_points(&img, &edges);
        for c in &found {
            draw_circle(&mut out, c.x as f32, c.y as f32, c.r, [0.0, 0.9, 1.0]);
        }
        save_rgb(&out, format!("{dir}/circles-{name}.png"));
    }

    // Synthetic nuclei (blurred, textured, slightly elongated), radii 6 to 16.
    let (nuc, labels) = samples::nuclei(160, 12, 7);
    let n_true = labels.pixels().iter().copied().max().unwrap_or(0) as usize;
    let mut centres = vec![(0.0f32, 0.0f32, 0.0f32); n_true + 1];
    for (x, y) in labels.coords() {
        let l = labels[(x, y)] as usize;
        centres[l] = (centres[l].0 + x as f32, centres[l].1 + y as f32, centres[l].2 + 1.0);
    }
    let centres: Vec<(f32, f32)> = centres[1..].iter().filter(|c| c.2 > 0.0).map(|c| (c.0 / c.2, c.1 / c.2)).collect();
    let edges = edge_points(&nuc, 1.5, 0.2);
    let radii: Vec<f32> = (6..=16).map(|r| r as f32).collect();
    for (name, mode, threshold) in [("full", None, 0.5), ("gradient", Some(Polarity::BrightOnDark), 0.25)] {
        let found = find_circles(&vote_all(&edges, 160, 160, &radii, mode), &radii, threshold);
        let hit = centres.iter().filter(|&&(x, y)| found.iter().any(|c| (c.x as f32 - x).hypot(c.y as f32 - y) <= 4.0)).count();
        let best = found.first().map_or(0.0, |c| c.score);
        println!("nuclei, {name} voting, threshold {threshold}: {} circles, {hit} of {} nuclei found, highest value {best:.2}", found.len(), centres.len());
        let mut out = show_points(&nuc, &edges);
        for c in &found {
            draw_circle(&mut out, c.x as f32, c.y as f32, c.r, [0.0, 0.9, 1.0]);
        }
        save_rgb(&out, format!("{dir}/nuclei-{name}.png"));
    }
    println!("wrote the pictures to {dir}/");
}
