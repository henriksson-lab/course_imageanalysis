// Checks for the coding exercises of "Fine-tuning strategies", using the numbers of the
// pen-and-paper exercises. Run: cargo test -p finetuning finetune_strategies::

use finetuning::finetune_strategies::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_pull_to_start() {
    let start: Net = vec![Layer { w: vec![vec![1.0, 2.0]], b: vec![0.0] }, Layer { w: vec![vec![1.0]], b: vec![0.0] }];
    let mut net = start.clone();
    net[0].w[0][0] = 1.5; // moved by 0.5
    net[1].w[0][0] = 4.0; // the head: no pull
    let mut g: Net = net.iter().map(|l| l.zeros_like()).collect();
    add_pull_to_start(&mut g, &net, &start, 0.1);
    assert!(close(g[0].w[0][0], 0.1, 1e-6), "2 λ (w − w0) = 2 · 0.1 · 0.5 = 0.1: got {}", g[0].w[0][0]);
    assert!(close(g[0].w[0][1], 0.0, 1e-6), "an unmoved weight gets no pull: got {}", g[0].w[0][1]);
    assert!(close(g[1].w[0][0], 0.0, 1e-6), "the last layer (the new head) is not pulled: got {}", g[1].w[0][0]);
}

#[test]
fn c2_param_counts() {
    assert_eq!(param_counts(32, 100, 3), (3200, 396), "32 × 100, rank 3");
    assert_eq!(param_counts(768, 768, 8), (589824, 12288), "768 × 768, rank 8");
}

#[test]
fn c3_c4_low_rank() {
    // a rank-1 change is fitted exactly by rank 1
    let u = [1.0f32, -2.0, 0.5];
    let v = [0.5f32, 1.0, 0.0, -1.0];
    let delta: Vec<Vec<f32>> = u.iter().map(|a| v.iter().map(|b| a * b).collect()).collect();
    let (b, a) = fit_low_rank(&delta, 1, 2000, 0.05, 1);
    assert!(b.len() == 3 && b[0].len() == 1 && a.len() == 1 && a[0].len() == 4, "B must be 3 × 1 and A 1 × 4");
    let e = rel_error(&delta, &b, &a);
    assert!(e < 1e-3, "rank 1 fits a rank-1 change: relative error {e}");
    // diag(3, 1): the best rank-1 error is 1 / sqrt(10) = 0.316
    let d = vec![vec![3.0, 0.0], vec![0.0, 1.0]];
    let (b, a) = fit_low_rank(&d, 1, 2000, 0.02, 2);
    let e = rel_error(&d, &b, &a);
    assert!(close(e, 0.3162, 2e-3), "diag(3, 1) at rank 1: relative error {e}, expected 1/sqrt(10) = 0.316");
    // C4 alone: B·A = 0 gives error 1
    let zero_b = vec![vec![0.0]; 2];
    let any_a = vec![vec![1.0, 1.0]];
    assert!(close(rel_error(&d, &zero_b, &any_a), 1.0, 1e-6), "B·A = 0 has relative error 1");
}
