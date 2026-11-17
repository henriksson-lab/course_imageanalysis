// Compares two simple segmentation methods on ten synthetic nucleus images with known
// truth: F1 at several thresholds τ, merges and splits, and a bootstrap over images of the
// difference in F1. Method "low": threshold 0.3 and connected components; method "high":
// threshold 0.6 and connected components. Results in out/cellnets-compare/.
//   cargo run -p day40 -- cellnets-compare                      (your code)
//   cargo run -p day40 --features solution -- cellnets-compare  (the model solution)

use crate::cellnets_compare::*;
use imgcourse::*;

const TAUS: [f32; 5] = [0.5, 0.6, 0.7, 0.8, 0.9];

/// Threshold, then label the 4-connected components; components smaller than 20 pixels are dropped.
fn threshold_and_label(img: &Image<f32>, level: f32) -> Image<u32> {
    let (w, h) = (img.width(), img.height());
    let mut labels = Image::new(w, h, 0u32);
    let mut next = 1;
    for (x, y) in img.coords() {
        if img[(x, y)] <= level || labels[(x, y)] != 0 {
            continue;
        }
        let mut stack = vec![(x, y)];
        let mut pixels = vec![(x, y)];
        labels[(x, y)] = next;
        while let Some((u, v)) = stack.pop() {
            for (du, dv) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                let (a, b) = (u as isize + du, v as isize + dv);
                if img.contains(a, b) && img[(a as usize, b as usize)] > level && labels[(a as usize, b as usize)] == 0 {
                    labels[(a as usize, b as usize)] = next;
                    stack.push((a as usize, b as usize));
                    pixels.push((a as usize, b as usize));
                }
            }
        }
        if pixels.len() < 20 {
            for p in pixels {
                labels[p] = u32::MAX; // marked as seen, removed below
            }
        } else {
            next += 1;
        }
    }
    labels.map(|l| if l == u32::MAX { 0 } else { l })
}

/// A 3 × 3 mean filter (the smoothing step of the pipeline).
fn smooth(img: &Image<f32>) -> Image<f32> {
    img.map_xy(|x, y, _| {
        let mut s = 0.0;
        for dy in -1..=1isize {
            for dx in -1..=1isize {
                s += img.get_border(x as isize + dx, y as isize + dy, Border::Clamp);
            }
        }
        s / 9.0
    })
}

pub fn main() {
    let dir = "out/cellnets-compare";
    let methods = [("low", 0.3), ("high", 0.6)];
    let mut f1 = [vec![], vec![]]; // per method, per image: F1 at every τ
    println!("image  nuclei | low: objects merges splits F1(0.5) F1(0.7) | high: objects merges splits F1(0.5) F1(0.7)");
    for seed in 1..=10u64 {
        let (img, truth) = samples::nuclei(256, 25, seed);
        let n_true = iou_table(&truth, &truth).len();
        let s = smooth(&img);
        let mut line = format!("{seed:5}  {n_true:6}");
        for (m, &(name, level)) in methods.iter().enumerate() {
            let pred = threshold_and_label(&s, level);
            let iou = iou_table(&truth, &pred);
            let n_pred = iou.first().map_or(0, |r| r.len());
            let curve = f1_curve(&iou, n_pred, &TAUS);
            let (merges, splits) = merges_and_splits(&overlaps(&truth, &pred));
            line += &format!(" | {n_pred:12} {merges:6} {splits:6} {:7.3} {:7.3}", curve[0], curve[2]);
            if seed == 1 {
                save_gray(&img, format!("{dir}/input.png"));
                save_labels(&truth, format!("{dir}/truth.png"));
                save_labels(&pred, format!("{dir}/{name}.png"));
            }
            f1[m].push(curve);
        }
        println!("{line}");
    }
    for (m, &(name, _)) in methods.iter().enumerate() {
        let mean: Vec<String> = (0..TAUS.len()).map(|k| format!("{:.3}", f1[m].iter().map(|c| c[k]).sum::<f32>() / f1[m].len() as f32)).collect();
        println!("mean F1({TAUS:?}) of {name:4}: {}", mean.join(", "));
    }
    for (k, tau) in [(0, 0.5), (2, 0.7)] {
        let d: Vec<f32> = (0..f1[0].len()).map(|i| f1[1][i][k] - f1[0][i][k]).collect();
        let (mean, lo, hi) = bootstrap_interval(&d, 10_000, 1);
        println!("F1 at τ = {tau}, high − low: mean {mean:+.3}, 95 % bootstrap interval [{lo:+.3}, {hi:+.3}]");
    }
    // The seven images of the lecture (exercise B6): StarDist − classical.
    let d = [0.042, 0.021, 0.431, 0.053, -0.015, 0.0, 0.0];
    let (mean, lo, hi) = bootstrap_interval(&d, 10_000, 1);
    println!("B6, all seven images: mean {mean:.3}, interval [{lo:.3}, {hi:.3}]");
    let (mean, lo, hi) = bootstrap_interval(&[0.042, 0.021, 0.053, -0.015, 0.0, 0.0], 10_000, 1);
    println!("B6, without the dense image: mean {mean:.3}, interval [{lo:.3}, {hi:.3}]");
    println!("wrote the results to {dir}/");
}
