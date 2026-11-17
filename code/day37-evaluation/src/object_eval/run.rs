// Scores two segmentations of a synthetic fluorescence image against its true nuclei:
// a threshold followed by connected-component labelling (touching nuclei merge), and the
// true nuclei drawn one pixel too large. Results in out/object-eval/.
//   cargo run -p day37 -- object-eval                      (your code)
//   cargo run -p day37 --features solution -- object-eval  (the model solution)

use imgcourse::*;
use crate::object_eval::*;

/// Connected components (4-neighbours) of a mask, labelled 1, 2, ... in the order they
/// are found, row by row. Components smaller than `min_size` pixels are dropped.
fn label_components(mask: &Image<bool>, min_size: usize) -> Image<u32> {
    let mut labels = Image::new(mask.width(), mask.height(), 0u32);
    let mut next = 1;
    for (x0, y0) in mask.coords() {
        if !mask[(x0, y0)] || labels[(x0, y0)] != 0 {
            continue;
        }
        // flood fill from (x0, y0)
        let mut stack = vec![(x0, y0)];
        let mut members = Vec::new();
        labels[(x0, y0)] = next;
        while let Some((x, y)) = stack.pop() {
            members.push((x, y));
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if mask.contains(nx, ny) && mask[(nx as usize, ny as usize)] && labels[(nx as usize, ny as usize)] == 0 {
                    labels[(nx as usize, ny as usize)] = next;
                    stack.push((nx as usize, ny as usize));
                }
            }
        }
        if members.len() < min_size {
            for (x, y) in members {
                labels[(x, y)] = 0;
            }
        } else {
            next += 1;
        }
    }
    labels
}

/// Every object grown by one pixel (into background pixels next to it).
fn grow(labels: &Image<u32>) -> Image<u32> {
    labels.map_xy(|x, y, l| {
        if l != 0 {
            return l;
        }
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let n = labels.get_border(x as isize + dx, y as isize + dy, Border::Zero);
            if n != 0 {
                return n;
            }
        }
        0
    })
}

/// The labels renumbered 1, 2, ... without gaps (a nucleus hidden under others disappears).
fn compact(labels: &Image<u32>) -> Image<u32> {
    let mut new = vec![0u32; count(labels) + 1];
    let mut next = 0;
    for &l in labels.pixels() {
        if l > 0 && new[l as usize] == 0 {
            next += 1;
            new[l as usize] = next;
        }
    }
    labels.map(|l| new[l as usize])
}

fn count(labels: &Image<u32>) -> usize {
    labels.pixels().iter().copied().max().unwrap_or(0) as usize
}

fn pixel_dice(a: &Image<u32>, b: &Image<u32>) -> f32 {
    let (mut both, mut sum) = (0, 0);
    for (x, y) in a.coords() {
        let (p, q) = (a[(x, y)] > 0, b[(x, y)] > 0);
        both += (p && q) as usize;
        sum += p as usize + q as usize;
    }
    2.0 * both as f32 / sum as f32
}

/// Matched predictions green, unmatched predictions (FP) red, missed true objects (FN) orange.
fn error_picture(truth: &Image<u32>, pred: &Image<u32>, tau: f32) -> Image<[f32; 3]> {
    let iou = iou_table(truth, pred);
    let matches = match_greedy(&iou, tau);
    let mut pred_hit = vec![false; count(pred)];
    let mut true_hit = vec![false; count(truth)];
    for &(i, j) in &matches {
        true_hit[i] = true;
        pred_hit[j] = true;
    }
    truth.map_xy(|x, y, g| {
        let p = pred[(x, y)];
        if p > 0 {
            if pred_hit[p as usize - 1] { [0.2, 0.75, 0.3] } else { [0.9, 0.15, 0.1] }
        } else if g > 0 && !true_hit[g as usize - 1] {
            [1.0, 0.6, 0.1]
        } else {
            [0.0, 0.0, 0.0]
        }
    })
}

pub fn main() {
    let dir = "out/object-eval";
    let (img, truth) = samples::nuclei(256, 30, 11);
    let truth = compact(&truth);
    save_gray(&img, format!("{dir}/input.png"));
    save_labels(&truth, format!("{dir}/truth.png"));

    let threshold = label_components(&img.map(|v| v > 0.4), 5);
    let larger = grow(&truth);
    save_labels(&threshold, format!("{dir}/threshold.png"));
    println!("true objects: {}", count(&truth));

    for (name, pred) in [("threshold + labelling", &threshold), ("truth, 1 px too large", &larger)] {
        let iou = iou_table(&truth, pred);
        println!("\n{name}: {} objects, pixel Dice {:.3}", count(pred), pixel_dice(&truth, pred));
        println!("  tau   TP  FP  FN  precision recall  F1");
        for tau in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9] {
            let s = scores(count(&truth), count(pred), match_greedy(&iou, tau).len());
            println!("  {tau:.1}  {:3} {:3} {:3}    {:.3}    {:.3}  {:.3}", s.tp, s.fp, s.fn_, s.precision, s.recall, s.f1);
        }
    }
    save_rgb(&error_picture(&truth, &threshold, 0.5), format!("{dir}/threshold-errors.png"));

    // Average precision of the threshold result, with each object's mean brightness as its
    // score: sort the predictions by score and mark each TP or FP at tau = 0.5.
    let n = count(&threshold);
    let mut brightness = vec![(0.0f32, 0usize); n];
    for (x, y) in img.coords() {
        let p = threshold[(x, y)] as usize;
        if p > 0 {
            brightness[p - 1].0 += img[(x, y)];
            brightness[p - 1].1 += 1;
        }
    }
    let iou = iou_table(&truth, &threshold);
    let matches = match_greedy(&iou, 0.5);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| (brightness[b].0 / brightness[b].1 as f32).total_cmp(&(brightness[a].0 / brightness[a].1 as f32)));
    let is_tp: Vec<bool> = order.iter().map(|&j| matches.iter().any(|&(_, m)| m == j)).collect();
    println!("\nthreshold, sorted by brightness: {}", is_tp.iter().map(|&t| if t { "TP" } else { "FP" }).collect::<Vec<_>>().join(" "));
    println!("average precision at tau = 0.5: {:.3}", average_precision(&is_tp, count(&truth)));
    println!("wrote the pictures to {dir}/");
}
