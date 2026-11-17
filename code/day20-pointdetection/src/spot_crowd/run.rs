// Evaluates a spot detector on synthetic images: greedy against optimal matching, F1 and RMSE
// against the gate and the density, and the precision–recall curve. Pictures in out/spot-crowd/.
//   cargo run -p day20 -- spot-crowd                      (your code)
//   cargo run -p day20 --features solution -- spot-crowd  (the model solution)

use imgcourse::*;
use crate::common::spot_scene;
use crate::spot_crowd::sim::{detect, draw, match_in_groups};
use crate::spot_crowd::*;

pub fn main() {
    let dir = "out/spot-crowd";
    // The matching demo's image: spots of σ = 1.5, amplitudes 8–80, noise 10 (grey levels / 255).
    let g = |v: f32| v / 255.0;
    let scene = |n: usize| spot_scene(256, n, 1.5, (g(8.0), g(80.0)), g(10.0), |_, _| g(40.0), 1);
    let positions = |d: &[(f32, f32, f32)]| d.iter().map(|&(x, y, _)| [x, y]).collect::<Vec<_>>();

    // 1. Density: 80, 300, 1000 spots; detector at k = 4, gate 2 px.
    println!("spots   found   TP    FP    FN   recall  precision   F1    RMSE (px)   greedy TP");
    for n in [80, 300, 1000] {
        let (img, truth) = scene(n);
        let det = positions(&detect(&img, 1.5, 4.0));
        let pairs = match_in_groups(&truth, &det, 2.0, match_optimal);
        let s = scores(&truth, &det, &pairs);
        let greedy = match_greedy(&truth, &det, 2.0).len();
        println!(
            "{n:>5} {:>7} {:>4} {:>5} {:>5}   {:>6.2} {:>10.2} {:>6.2} {:>9.2} {:>11}",
            det.len(), s.tp, s.fp, s.fn_, s.recall, s.precision, s.f1, s.rmse, greedy
        );
        save_rgb(&draw(&img, &truth, &det, &pairs), format!("{dir}/matching-{n}.png"));
    }

    // 2. The gate: 80 spots, k = 4.
    let (img, truth) = scene(80);
    let det = positions(&detect(&img, 1.5, 4.0));
    println!("gate (px)   TP   FP   FN    F1    RMSE");
    for gate in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0] {
        let s = scores(&truth, &det, &match_in_groups(&truth, &det, gate, match_optimal));
        println!("{gate:>9.1} {:>4} {:>4} {:>4} {:>5.2} {:>7.2}", s.tp, s.fp, s.fn_, s.f1, s.rmse);
    }

    // 3. Precision–recall: all candidates down to k = 2, ranked by score, matched once.
    let ranked = detect(&img, 1.5, 2.0);
    let det = positions(&ranked);
    let pairs = match_in_groups(&truth, &det, 2.0, match_optimal);
    let mut is_tp = vec![false; det.len()];
    for &(_, j) in &pairs {
        is_tp[j] = true;
    }
    let (curve, ap) = pr_curve(&is_tp, truth.len());
    let (best, &(p, r)) = curve
        .iter()
        .enumerate()
        .max_by(|a, b| {
            let f = |c: &(f32, f32)| 2.0 * c.0 * c.1 / (c.0 + c.1).max(1e-9);
            f(a.1).total_cmp(&f(b.1))
        })
        .unwrap();
    println!(
        "PR curve over {} candidates: AP {ap:.2}; best F1 {:.2} (precision {p:.2}, recall {r:.2}) keeping the {} strongest, k = {:.1}",
        det.len(),
        2.0 * p * r / (p + r),
        best + 1,
        ranked[best].2
    );
    // the curve as a small picture: recall across, precision up
    let mut plot = Image::new(201, 201, [1.0f32, 1.0, 1.0]);
    for &(p, r) in &curve {
        let (x, y) = ((r * 200.0).round() as usize, ((1.0 - p) * 200.0).round() as usize);
        plot[(x.min(200), y.min(200))] = [0.1, 0.3, 0.9];
    }
    save_rgb(&plot, format!("{dir}/pr-curve.png"));
    println!("wrote the pictures to {dir}/ (rings = truth: white matched, yellow missed; dots = detections: green matched, red false)");
}
