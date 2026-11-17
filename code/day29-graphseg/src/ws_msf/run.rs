// The seeded watershed as a minimum spanning forest on synthetic nuclei: one marker per
// nucleus and one for the background, on the gradient landscape and on |ΔI|. Pictures in
// out/ws-msf/.
//   cargo run -p day29 -- ws-msf                      (your code)
//   cargo run -p day29 --features solution -- ws-msf  (the model solution)

use imgcourse::*;
use std::time::Instant;
use crate::ws_msf::*;
use crate::common::gradient;

/// For every true nucleus, the best overlap (intersection over union) with a region of `labels`.
fn matched(labels: &[u32], truth: &Image<u32>, n: u32) -> usize {
    (1..=n)
        .filter(|&t| {
            let mine: Vec<usize> = (0..labels.len()).filter(|&p| truth.pixels()[p] == t).collect();
            if mine.is_empty() {
                return false;
            }
            // the region that covers most of the nucleus
            let mut counts = std::collections::HashMap::new();
            for &p in &mine {
                *counts.entry(labels[p]).or_insert(0usize) += 1;
            }
            let (&l, &inter) = counts.iter().max_by_key(|(_, c)| **c).unwrap();
            let region = labels.iter().filter(|&&v| v == l).count();
            inter as f32 / (mine.len() + region - inter) as f32 >= 0.5
        })
        .count()
}

pub fn main() {
    let dir = "out/ws-msf";
    let (img, truth) = samples::nuclei(256, 18, 4);
    let (w, h) = (img.width(), img.height());
    let n_nuclei = truth.pixels().iter().copied().max().unwrap_or(0);
    save_gray(&img, format!("{dir}/nuclei.png"));

    // Markers: background = the image border (label 1); nucleus k = a 3 × 3 square at its
    // centre (label k + 1).
    let mut markers = vec![0u32; w * h];
    for x in 0..w {
        markers[x] = 1;
        markers[(h - 1) * w + x] = 1;
    }
    for y in 0..h {
        markers[y * w] = 1;
        markers[y * w + w - 1] = 1;
    }
    for l in 1..=n_nuclei {
        let pix: Vec<(usize, usize)> = truth.coords().filter(|&p| truth[p] == l).collect();
        if pix.is_empty() {
            continue;
        }
        let cx = pix.iter().map(|p| p.0).sum::<usize>() / pix.len();
        let cy = pix.iter().map(|p| p.1).sum::<usize>() / pix.len();
        for dy in 0..3 {
            for dx in 0..3 {
                markers[(cy + dy - 1) * w + cx + dx - 1] = l + 1;
            }
        }
    }

    let landscape = gradient(&img);
    save_gray_stretched(&landscape, format!("{dir}/landscape.png"));
    let start = Instant::now();
    let edges = altitude_edges(&landscape);
    let (labels, cuts) = prim_from_markers(w * h, &edges, &markers);
    let t_prim = start.elapsed().as_secs_f64();
    let nuclei_labels: Vec<u32> = labels.iter().map(|&l| l - 1).collect(); // background 0
    println!("{w} × {h}, {n_nuclei} nuclei, one marker each plus the border:");
    println!("  Prim on the gradient: {} cut edges, {:.0} ms; {} of {n_nuclei} nuclei matched (IoU ≥ 0.5)", cuts.len(), 1e3 * t_prim, matched(&nuclei_labels, &truth, n_nuclei));
    save_labels(&Image::from_vec(w, h, nuclei_labels.clone()), format!("{dir}/watershed.png"));

    let start = Instant::now();
    let k = kruskal_from_markers(w * h, &edges, &markers);
    let same = k.iter().zip(&labels).filter(|(a, b)| a == b).count();
    println!("  Kruskal with a virtual root: {:.0} ms; same label as Prim at {same} of {} pixels", 1e3 * start.elapsed().as_secs_f64(), w * h);

    // The same markers with |ΔI| of the image as edge weights instead of a landscape.
    let diff: Vec<Edge> = edges.iter().map(|&(a, b, _)| (a, b, (img.pixels()[a] - img.pixels()[b]).abs())).collect();
    let (labels_diff, _) = prim_from_markers(w * h, &diff, &markers);
    let labels_diff: Vec<u32> = labels_diff.iter().map(|&l| l - 1).collect();
    println!("  Prim on |ΔI| of the image: {} of {n_nuclei} nuclei matched", matched(&labels_diff, &truth, n_nuclei));
    save_labels(&Image::from_vec(w, h, labels_diff), format!("{dir}/watershed-diff.png"));
    save_labels(&truth, format!("{dir}/truth.png"));
    println!("\nwrote the pictures to {dir}/");
}
