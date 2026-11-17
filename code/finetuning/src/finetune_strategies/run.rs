// Fine-tuning strategies on the toy patches: frozen layers, one learning rate or one per layer,
// a pull back to the pretrained weights, and how much of the old task survives (forgetting).
// Then the change that fine-tuning made to layer 1, fitted by a low-rank B·A.
//   cargo run -p finetuning -- finetune-strategies                      (your code)
//   cargo run -p finetuning --features solution -- finetune-strategies  (the model solution)

use crate::finetune_strategies::*;
use imgcourse::linalg;
use imgcourse::*;

/// Full-batch gradient descent with per-layer rates and an optional pull to the start weights.
fn tune(start: &Net, xs: &[Vec<f32>], ys: &[usize], rates: &[f32], lambda: f32, steps: usize) -> Net {
    let mut net = start.clone();
    let all: Vec<usize> = (0..xs.len()).collect();
    for _ in 0..steps {
        let (mut g, _) = gradients(&net, xs, ys, &all);
        if lambda > 0.0 {
            add_pull_to_start(&mut g, &net, start, lambda);
        }
        descend(&mut net, &g, rates);
    }
    net
}

/// The old-task accuracy of a backbone with the old head put back.
fn old_task(net: &Net, pre: &Net, xs: &[Vec<f32>], ys: &[usize]) -> f32 {
    let mut old = net.clone();
    let last = old.len() - 1;
    old[last] = pre[last].clone();
    net_accuracy(&old, xs, ys)
}

pub fn main() {
    let dir = "out/finetune-strategies";
    println!("pretraining on the old task...");
    let pre = pretrained();
    let (old_x, old_y) = make_set(true, 400, 202, LOOK_A);
    let (pool_x, pool_y) = make_set(false, 300, 303, LOOK_A);
    let (val_x, val_y) = make_set(false, 300, 404, LOOK_A);
    let start = with_new_head(&pre, 3, 11);
    let lr = 0.05;
    println!("old task before fine-tuning: {:.1} %", 100.0 * old_task(&start, &pre, &old_x, &old_y));
    for n in [9, 300] {
        println!("\n{n} labels, 500 steps: new task / old task (old head put back)");
        let rows: [(&str, [f32; 3], f32); 5] = [
            ("head only", [0.0, 0.0, lr], 0.0),
            ("last layers", [0.0, lr, lr], 0.0),
            ("all layers, one rate", [lr, lr, lr], 0.0),
            ("all layers, rates /10 /3 /1", [lr / 10.0, lr / 3.0, lr], 0.0),
            ("all layers, pull to start", [lr, lr, lr], 0.2),
        ];
        for (name, rates, lambda) in rows {
            let net = tune(&start, &pool_x[..n], &pool_y[..n], &rates, lambda, 500);
            println!(
                "  {name:28} {:5.1} %  {:5.1} %",
                100.0 * net_accuracy(&net, &val_x, &val_y),
                100.0 * old_task(&net, &pre, &old_x, &old_y)
            );
        }
    }

    // the change of layer 1 made by fine-tuning all layers on 300 labels
    let tuned = tune(&start, &pool_x, &pool_y, &[lr, lr, lr], 0.0, 500);
    let delta: Vec<Vec<f32>> = (0..32).map(|o| (0..P * P).map(|i| tuned[0].w[o][i] - pre[0].w[o][i]).collect()).collect();
    // singular values: square roots of the eigenvalues of Δ Δᵀ (32 × 32)
    let ddt: Vec<Vec<f64>> = (0..32).map(|i| (0..32).map(|j| (0..P * P).map(|k| (delta[i][k] * delta[j][k]) as f64).sum()).collect()).collect();
    let (ev, _) = linalg::sym_eigen(&ddt);
    let sv: Vec<f64> = ev.iter().map(|v| v.max(0.0).sqrt()).collect();
    let total: f64 = sv.iter().map(|s| s * s).sum();
    println!("\nchange of layer 1 (32 × 100) after fine-tuning on 300 labels");
    println!("  largest singular values: {}", sv[..8].iter().map(|s| format!("{s:.2}")).collect::<Vec<_>>().join(" "));
    println!("  rank   numbers (full 3200)   best possible error   gradient descent error   new-task accuracy");
    for r in [1, 2, 3, 4, 8] {
        let (b, a) = fit_low_rank(&delta, r, 3000, 0.01, 5);
        let best = (sv[r..].iter().map(|s| s * s).sum::<f64>() / total).sqrt();
        let mut net = tuned.clone();
        let ba = matmul(&b, &a);
        for o in 0..32 {
            for i in 0..P * P {
                net[0].w[o][i] = pre[0].w[o][i] + ba[o][i];
            }
        }
        let (_, low) = param_counts(32, P * P, r);
        println!(
            "  {r:4}   {low:19}   {:18.1} %   {:21.1} %   {:15.1} %",
            100.0 * best,
            100.0 * rel_error(&delta, &b, &a),
            100.0 * net_accuracy(&net, &val_x, &val_y)
        );
    }
    println!("  (the network keeps its fine-tuned layer 2 and head; full change: {:.1} %)", 100.0 * net_accuracy(&tuned, &val_x, &val_y));
    let tiles: Vec<Image<f32>> = delta.iter().map(|row| filter_image(row)).collect();
    save_gray(&mosaic(&tiles, 8, 8), format!("{dir}/delta-filters.png"));
    println!("wrote {dir}/delta-filters.png (the change of each first-layer filter, grey = 0)");
}
