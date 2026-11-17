// A toy of contrastive learning: 64 synthetic cells (round, elongated, a dividing pair; any
// brightness), a tiny encoder 256 → 16 → 2 normalised onto the unit circle, trained with
// InfoNCE (τ = 0.2, both directions) on two augmented views of every cell per step, 600 steps.
// The labels are never used for training: they score the embedding afterwards (5-NN accuracy).
// Pictures of the embeddings in out/contrastive/.
//   cargo run -p selfsup -- contrastive                      (your code)
//   cargo run -p selfsup --features solution -- contrastive  (the model solution)

use crate::common::{augment, knn_accuracy, make_cells, unit, Adam, Augment, Cell, Mlp, CELL};
use crate::contrastive::*;
use imgcourse::{save_rgb, Image, Rng};

const STEPS: usize = 600;

/// Train one encoder; `negatives = false` only pulls the two views together.
fn train(cells: &[Cell], aug: Augment, tau: f32, negatives: bool) -> (Mlp, Vec<f32>) {
    let mut net = Mlp::new(CELL * CELL, 16, 2, 1);
    let mut adam = Adam::new(&net, 0.01);
    let mut rng = Rng::new(1000);
    let mut losses = Vec::new();
    for _ in 0..STEPS {
        let pa: Vec<_> = cells.iter().map(|c| net.forward(&augment(&c.pixels, aug, &mut rng))).collect();
        let pb: Vec<_> = cells.iter().map(|c| net.forward(&augment(&c.pixels, aug, &mut rng))).collect();
        let a: Vec<Vec<f32>> = pa.iter().map(|p| unit(&p.out)).collect();
        let b: Vec<Vec<f32>> = pb.iter().map(|p| unit(&p.out)).collect();
        let n = a.len() as f32;
        let (loss, ga, gb) = if negatives {
            // both directions, averaged: a picks its partner among all b, and b among all a
            let (l1, _) = info_nce(&a, &b, tau);
            let (l2, _) = info_nce(&b, &a, tau);
            let (ga1, gb1) = info_nce_grad(&a, &b, tau);
            let (gb2, ga2) = info_nce_grad(&b, &a, tau);
            let half = |x: &[Vec<f32>], y: &[Vec<f32>]| -> Vec<Vec<f32>> { x.iter().zip(y).map(|(u, v)| u.iter().zip(v).map(|(p, q)| 0.5 * (p + q)).collect()).collect() };
            (0.5 * (l1 + l2), half(&ga1, &ga2), half(&gb1, &gb2))
        } else {
            let l = -a.iter().zip(&b).map(|(u, v)| dot(u, v)).sum::<f32>() / n;
            (l, b.iter().map(|v| v.iter().map(|x| -x / n).collect()).collect(), a.iter().map(|v| v.iter().map(|x| -x / n).collect()).collect())
        };
        let mut g = net.zeros_like();
        for (p, gz) in pa.iter().zip(&ga).chain(pb.iter().zip(&gb)) {
            net.backward(p, &unit_backward(&p.out, gz), &mut g);
        }
        adam.step(&mut net, &mut g);
        losses.push(loss);
    }
    (net, losses)
}

/// The embeddings on the circle, coloured by shape (blue round, red elongated, green pair).
fn picture(z: &[Vec<f32>], cells: &[Cell]) -> Image<[f32; 3]> {
    let s = 220;
    let mut img = Image::new(s, s, [1.0f32; 3]);
    for k in 0..720 {
        let t = k as f32 * std::f32::consts::PI / 360.0;
        let (x, y) = (110.0 + 90.0 * t.cos(), 110.0 - 90.0 * t.sin());
        img[(x as usize, y as usize)] = [0.7; 3];
    }
    let colours = [[0.12, 0.47, 0.71], [0.84, 0.15, 0.16], [0.17, 0.63, 0.17]];
    for (q, c) in z.iter().zip(cells) {
        let (cx, cy) = (110.0 + 90.0 * q[0], 110.0 - 90.0 * q[1]);
        for dy in -3..=3 {
            for dx in -3..=3 {
                let (x, y) = (cx as isize + dx, cy as isize + dy);
                if dx * dx + dy * dy <= 9 && img.contains(x, y) {
                    img[(x as usize, y as usize)] = colours[c.shape];
                }
            }
        }
    }
    img
}

pub fn main() {
    let dir = "out/contrastive";
    let cells = make_cells(64, 1);
    let full = Augment { strength: 1.0, geometry: true, brightness: true, noise: true };
    let shapes: Vec<usize> = cells.iter().map(|c| c.shape).collect();
    let dim: Vec<usize> = cells.iter().map(|c| usize::from(c.bright >= 0.675)).collect();
    println!("64 cells, encoder 256 -> 16 -> 2 (unit circle), tau 0.2, {STEPS} steps; ln 64 = {:.3}", 64f32.ln());
    println!("{:<34} {:>11} {:>11} {:>8} {:>8} {:>9}", "", "loss start", "loss end", "shape", "bright", "|mean z|");
    for (name, aug, negatives) in [
        ("all augmentations", full, true),
        ("no brightness augmentation", Augment { brightness: false, ..full }, true),
        ("no augmentation", Augment { strength: 0.0, ..full }, true),
        ("all augmentations, no negatives", full, false),
    ] {
        let (net, losses) = train(&cells, aug, 0.2, negatives);
        let z: Vec<Vec<f32>> = cells.iter().map(|c| unit(&net.forward(&c.pixels).out)).collect();
        let mean = [z.iter().map(|q| q[0]).sum::<f32>() / 64.0, z.iter().map(|q| q[1]).sum::<f32>() / 64.0];
        let spread = (mean[0] * mean[0] + mean[1] * mean[1]).sqrt();
        let (sa, ba) = (knn_accuracy(&z, &shapes, 3, 5), knn_accuracy(&z, &dim, 2, 5));
        let avg = |r: std::ops::Range<usize>| losses[r.clone()].iter().sum::<f32>() / r.len() as f32;
        println!("{name:<34} {:>11.3} {:>11.3} {:>7.0}% {:>7.0}% {spread:>9.2}", avg(0..10), avg(STEPS - 50..STEPS), 100.0 * sa, 100.0 * ba);
        save_rgb(&picture(&z, &cells), format!("{dir}/{}.png", name.replace([' ', ','], "-").replace("--", "-")));
    }
    println!("(loss: mean of the first 10 and of the last 50 steps; shape and bright: 5-NN accuracy on the");
    println!(" embeddings of the clean cells, chance 33 % and 50 %; |mean z| = 1: all cells in one point)");
    println!("wrote the pictures to {dir}/");
}
