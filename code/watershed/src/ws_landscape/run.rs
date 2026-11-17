// Floods 1D profiles and draws them; pictures in out/ws-landscape/.
//   cargo run -p watershed -- ws-landscape                      (your code)
//   cargo run -p watershed --features solution -- ws-landscape  (the model solution)

use imgcourse::*;
use crate::ws_landscape::*;

/// Draw a profile as a landscape: one column of 6 pixels per sample, filled up to its height
/// in the colour of its basin; dams black, sources marked by a white dot at the bottom.
fn draw(f: &[f32], labels: &[u32], sources: &[u32], path: &str) {
    let (lo, hi) = f.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let height = 160;
    let colours = io::label_colours(&Image::from_vec(f.len(), 1, labels.to_vec()));
    let dam_at: Vec<usize> = dams(f, labels).iter().map(|d| d.0).collect();
    let img = Image::from_fn(6 * f.len(), height, |x, y| {
        let i = x / 6;
        let top = height - 10 - ((f[i] - lo) / (hi - lo).max(1e-6) * (height - 30) as f32) as usize;
        if y >= height - 4 {
            if sources[i] > 0 { [1.0; 3] } else { [0.15; 3] }
        } else if y >= top {
            if dam_at.contains(&i) { [0.0; 3] } else { colours[(i, 0)] }
        } else {
            [0.92; 3]
        }
    });
    save_rgb(&img, path);
}

fn count(labels: &[u32]) -> u32 {
    labels.iter().copied().max().unwrap_or(0)
}

pub fn main() {
    let dir = "out/ws-landscape";

    // 1. "Two cells": a line across two dark nuclei on a bright background, then with noise.
    let n = 64;
    let clean: Vec<f32> = (0..n)
        .map(|i| {
            let x = i as f32;
            8.0 - 5.6 * (-((x - 20.0) / 8.0).powi(2)).exp() - 4.6 * (-((x - 42.0) / 8.0).powi(2)).exp()
        })
        .collect();
    let mut rng = Rng::new(2);
    let noisy: Vec<f32> = clean.iter().map(|v| v + 0.25 * rng.normal()).collect();
    for (name, f) in [("clean", &clean), ("noisy", &noisy)] {
        let m = regional_minima(f);
        let labels = flood(f, &m);
        let d: Vec<String> = dams(f, &labels).iter().map(|(i, h)| format!("{i} ({h:.1})")).collect();
        println!("two cells, {name}: {} minima, {} regions; dams at {}", count(&m), count(&labels), d.join(", "));
        draw(f, &labels, &m, &format!("{dir}/two-cells-{name}.png"));
    }
    print!("noisy, minima left after h-minima:");
    for h in [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0] {
        print!("  h = {h}: {}", count(&regional_minima(&h_minima(&noisy, h))));
    }
    println!();
    let m = regional_minima(&h_minima(&noisy, 1.0));
    let labels = flood(&noisy, &m);
    let d: Vec<String> = dams(&noisy, &labels).iter().map(|(i, h)| format!("{i} ({h:.1})")).collect();
    println!("noisy, flooded from the h-minima with h = 1: {} regions, dams at {}", count(&labels), d.join(", "));
    draw(&noisy, &labels, &m, &format!("{dir}/two-cells-noisy-h1.png"));

    // 2. "Edges": three flat levels with noise. Flood f, and flood |f'|.
    let mut rng = Rng::new(5);
    let step: Vec<f32> = (0..60).map(|i| if i < 20 { 1.0 } else if i < 40 { 5.0 } else { 2.5 }).map(|v| v + 0.1 * rng.normal()).collect();
    let deriv = abs_derivative(&step);
    for (name, landscape) in [("f", &step), ("derivative", &deriv)] {
        let m = regional_minima(&h_minima(landscape, 0.3));
        let labels = flood(landscape, &m);
        let d: Vec<String> = dams(landscape, &labels).iter().map(|(i, _)| format!("{i}")).collect();
        println!("edges, flood {name} (h = 0.3): {} regions, dams at samples {}", count(&labels), d.join(", "));
        draw(&step, &labels, &m, &format!("{dir}/edges-flood-{name}.png"));
    }
    println!("wrote the results to {dir}/");
}
