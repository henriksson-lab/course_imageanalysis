// The pieces of self-distillation, on the toy of the contrastive unit (64 synthetic cells):
//  1. what centring and sharpening do to a batch of teacher outputs: the teacher's 8 output
//     scores ("prototypes") for the 64 cells, from an untrained network, with a common drift
//     towards prototype 0 added (the beginning of a collapse);
//  2. the moving-average teacher: how far it follows a student that jumps.
//   cargo run -p selfsup -- self-distillation                      (your code)
//   cargo run -p selfsup --features solution -- self-distillation  (the model solution)

use crate::common::{make_cells, Mlp, CELL};
use crate::self_distillation::*;

const K: usize = 8;

pub fn main() {
    let cells = make_cells(64, 1);
    let net = Mlp::new(CELL * CELL, 16, K, 1);

    // 1. centring and sharpening
    let logits: Vec<Vec<f32>> = cells
        .iter()
        .map(|c| {
            let mut z = net.forward(&c.pixels).out;
            z[0] += 1.0; // a drift that every cell shares
            z
        })
        .collect();
    let mean: Vec<f32> = (0..K).map(|k| logits.iter().map(|z| z[k]).sum::<f32>() / 64.0).collect();
    let zero = vec![0.0; K];
    println!("1. A batch of teacher outputs (64 cells, {K} prototypes; uniform entropy ln {K} = {:.3})", (K as f32).ln());
    println!("   {:<32} {:>16} {:>18} {:>9}", "", "mean H per cell", "H of batch mean", "used");
    for (name, centre, tau) in [("plain softmax (tau 1)", &zero, 1.0), ("sharpened only (tau 0.04)", &zero, 0.04), ("centred only (tau 1)", &mean, 1.0), ("centred and sharpened", &mean, 0.04)] {
        let p: Vec<Vec<f32>> = logits.iter().map(|z| softmax_t(z, centre, tau)).collect();
        let h_each = p.iter().map(|q| entropy(q)).sum::<f32>() / 64.0;
        let avg: Vec<f32> = (0..K).map(|k| p.iter().map(|q| q[k]).sum::<f32>() / 64.0).collect();
        let mut used = [false; K];
        for q in &p {
            let best = (0..K).max_by(|&a, &b| q[a].total_cmp(&q[b])).unwrap_or(0);
            used[best] = true;
        }
        println!("   {name:<32} {h_each:>16.3} {:>18.3} {:>9}", entropy(&avg), used.iter().filter(|&&u| u).count());
    }
    println!("   (H per cell small = confident; H of the batch mean small = all cells in the same prototype)");

    // 2. the moving-average teacher follows a student weight that jumps from 0 to 1
    println!("\n2. A student weight jumps from 0 to 1; the teacher's copy after n steps:");
    for m in [0.9f32, 0.99, 0.996] {
        let mut teacher = [0.0f32];
        let mut row = format!("   m = {m:<6}");
        for n in 1..=1000 {
            ema(&mut teacher, &[1.0], m);
            if [1, 10, 100, 1000].contains(&n) {
                row += &format!("  n = {n}: {:.3}", teacher[0]);
            }
        }
        println!("{row}");
    }
}
