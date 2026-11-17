// Graph-cut segmentation of a noisy disk with a thin arm (thresholding against the cut, λ,
// 4 against 8 neighbours), and of synthetic nuclei from a few seeds. Pictures in
// out/graphcut-seg/.
//   cargo run -p graphcut-seg                      (your code)
//   cargo run -p graphcut-seg --features solution  (the model solution)

use graphcut_seg::*;
use imgcourse::*;

const SIZE: usize = 128;

/// The true labels: a disk of radius 30 and an arm 3 pixels wide running up to the right.
fn disk_with_arm() -> Image<u32> {
    Image::from_fn(SIZE, SIZE, |x, y| {
        let (dx, dy) = (x as f32 - 54.0, y as f32 - 70.0);
        let along = (dx - dy) / 2f32.sqrt();
        let across = (dx + dy) / 2f32.sqrt();
        let disk = dx * dx + dy * dy <= 30.0 * 30.0;
        let arm = along > 0.0 && along < 85.0 && across.abs() <= 1.5;
        if disk || arm { 1 } else { 0 }
    })
}

fn wrong(a: &Image<u32>, b: &Image<u32>) -> usize {
    a.pixels().iter().zip(b.pixels()).filter(|(p, q)| p != q).count()
}

/// The grey image in colour with the boundary of the object in yellow and the seeds
/// (1 object: red, 2 background: blue).
fn overlay(img: &Image<f32>, labels: &Image<u32>, seeds: Option<&Image<u32>>) -> Image<[f32; 3]> {
    let mut out = gray_to_rgb(&img.map(|v| v.clamp(0.0, 1.0)));
    for (x, y) in img.coords() {
        let l = labels[(x, y)];
        let edge = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            .iter()
            .any(|&(dx, dy)| labels.get_border(x as isize + dx, y as isize + dy, Border::Clamp) != l);
        if l == 1 && edge {
            out[(x, y)] = [1.0, 0.9, 0.0];
        }
        if let Some(s) = seeds {
            match s[(x, y)] {
                1 => out[(x, y)] = [1.0, 0.1, 0.1],
                2 => out[(x, y)] = [0.2, 0.4, 1.0],
                _ => {}
            }
        }
    }
    out
}

fn main() {
    let dir = "out/graphcut-seg";
    let truth = disk_with_arm();
    let (mu0, mu1) = (80.0 / 255.0, 170.0 / 255.0);
    let inf = f32::INFINITY;

    // 1. The noisy disk (noise 50 grey levels), known model: thresholding against the cut.
    let s = 50.0 / 255.0;
    let img = samples::add_noise(&truth.map(|l| if l == 1 { mu1 } else { mu0 }), s, 3);
    let d0 = img.map(|v| (v - mu0).powi(2) / (2.0 * s * s));
    let d1 = img.map(|v| (v - mu1).powi(2) / (2.0 * s * s));
    let thr = d0.zip_map(&d1, |a, b| if b < a { 1 } else { 0 });
    save_gray(&img, format!("{dir}/disk-input.png"));
    save_gray(&thr.map(|l| l as f32), format!("{dir}/disk-threshold.png"));
    println!("noisy disk with a thin arm, {SIZE} × {SIZE}, noise 50: wrong pixels");
    println!("  thresholding: {}", wrong(&thr, &truth));
    for lambda in [0.5, 1.5, 4.0, 10.0] {
        let start = std::time::Instant::now();
        let (labels, flow) = graph_cut(&img, &d0, &d1, lambda, inf, Neighbours::Four);
        println!(
            "  graph cut, 4 neighbours, λ = {lambda:4}: {:5} wrong, {:5} object pixels (truth {}), E = max flow = {flow:.1}  ({:.2} s)",
            wrong(&labels, &truth),
            labels.pixels().iter().sum::<u32>(),
            truth.pixels().iter().sum::<u32>(),
            start.elapsed().as_secs_f32()
        );
        save_gray(&labels.map(|l| l as f32), format!("{dir}/disk-cut-lambda{lambda}.png"));
    }

    // 2. Metrication: much more noise (90) and a large λ, 4 against 8 neighbours.
    let s = 90.0 / 255.0;
    let img = samples::add_noise(&truth.map(|l| if l == 1 { mu1 } else { mu0 }), s, 5);
    let d0 = img.map(|v| (v - mu0).powi(2) / (2.0 * s * s));
    let d1 = img.map(|v| (v - mu1).powi(2) / (2.0 * s * s));
    println!("\nnoise 90, λ = 4: wrong pixels");
    for (name, nb) in [("4", Neighbours::Four), ("8", Neighbours::Eight)] {
        let (labels, _) = graph_cut(&img, &d0, &d1, 4.0, inf, nb);
        println!("  {name} neighbours: {}", wrong(&labels, &truth));
        save_gray(&labels.map(|l| l as f32), format!("{dir}/blocky-{name}.png"));
    }

    // 3. Synthetic nuclei, with extra noise, and seeds: one object stroke in nucleus 1 and
    // background strokes in five places; the model comes from the seed pixels.
    let (clean, true_labels) = samples::nuclei(SIZE, 14, 2);
    let img = samples::add_noise(&clean, 0.15, 7);
    let nuclei = true_labels.map(|l| if l > 0 { 1 } else { 0 });
    let mut seeds = Image::new(SIZE, SIZE, 0u32);
    let (mut cx, mut cy, mut n) = (0, 0, 0);
    for (x, y) in true_labels.coords() {
        if true_labels[(x, y)] == 1 {
            cx += x;
            cy += y;
            n += 1;
        }
    }
    let (cx, cy) = (cx / n, cy / n);
    for d in 0..5 {
        seeds[(cx - 2 + d, cy)] = 1; // a short stroke across the middle of nucleus 1
    }
    for &(x0, y0) in &[(2, 2), (120, 3), (3, 122), (122, 122), (64, 2)] {
        for d in 0..4 {
            if nuclei[(x0 + d, y0)] == 0 {
                seeds[(x0 + d, y0)] = 2;
            }
        }
    }
    let (m1, s1) = seed_model(&img, &seeds, 1);
    let (m0, s0) = seed_model(&img, &seeds, 2);
    println!("\nsynthetic nuclei with noise 0.15: object seeds mean {m1:.2} sd {s1:.3}, background seeds mean {m0:.2} sd {s0:.3}");
    let (d0, d1) = seeded_costs(&img, &seeds);
    let thr = d0.zip_map(&d1, |a, b| if b < a { 1 } else { 0 });
    println!("  thresholding with the seed model: {} wrong pixels", wrong(&thr, &nuclei));
    for (lambda, sigma) in [(1.0, inf), (2.0, inf), (5.0, inf), (2.0, 0.4), (2.0, 0.1)] {
        let (labels, _) = graph_cut(&img, &d0, &d1, lambda, sigma, Neighbours::Eight);
        let found = (1..=14).filter(|&k| true_labels.coords().any(|(x, y)| true_labels[(x, y)] == k && labels[(x, y)] == 1)).count();
        println!("  graph cut, λ = {lambda}, σ = {sigma}: {} wrong pixels, {found} of 14 nuclei touched", wrong(&labels, &nuclei));
        if lambda == 2.0 && sigma == inf {
            save_rgb(&overlay(&img, &labels, Some(&seeds)), format!("{dir}/nuclei-cut.png"));
        }
    }
    save_rgb(&overlay(&img, &thr, Some(&seeds)), format!("{dir}/nuclei-threshold.png"));
    println!("\nwrote the pictures to {dir}/ (white = object; nuclei: boundary yellow, seeds red and blue)");
}
