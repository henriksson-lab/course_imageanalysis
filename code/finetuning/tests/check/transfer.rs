// Checks for the coding exercises of "Transfer learning", using the numbers of the
// pen-and-paper exercises. Run: cargo test -p finetuning transfer::

use finetuning::transfer::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// A 2 → 2 → 1 network with known weights.
fn tiny() -> Net {
    vec![
        Layer { w: vec![vec![1.0, -1.0], vec![0.5, 0.5]], b: vec![0.0, -1.0] },
        Layer { w: vec![vec![1.0, 1.0]], b: vec![0.0] },
    ]
}

#[test]
fn c1_features() {
    let f = features(&tiny(), &[2.0, 1.0], 1);
    assert!(f.len() == 2 && close(f[0], 1.0, 1e-6) && close(f[1], 0.5, 1e-6), "features after layer 1 of (2, 1): got {f:?}, expected [1, 0.5]");
    let f = features(&tiny(), &[1.0, 2.0], 1);
    assert!(close(f[0], 0.0, 1e-6) && close(f[1], 0.5, 1e-6), "the ReLU must cut negative values: got {f:?}, expected [0, 0.5]");
    let f = features(&tiny(), &[1.0, 2.0], 0);
    assert_eq!(f, vec![1.0, 2.0], "zero layers: the input itself");
}

#[test]
fn c2_train_probe_one_step() {
    // B: two examples, one feature: f = 0 (class 0) and f = 1 (class 1); one step with lr 1
    let probe = train_probe(&[vec![0.0], vec![1.0]], &[0, 1], 2, 1, 1.0);
    assert!(close(probe.w[0][0], -0.25, 1e-6) && close(probe.w[1][0], 0.25, 1e-6), "weights after one step: got {:?}, expected [[-0.25], [0.25]]", probe.w);
    assert!(close(probe.b[0], 0.0, 1e-6) && close(probe.b[1], 0.0, 1e-6), "biases after one step: got {:?}, expected [0, 0]", probe.b);
}

#[test]
fn c2_c3_probe_separates() {
    let feats = vec![vec![0.0, 1.0], vec![0.2, 0.9], vec![1.0, 0.0], vec![0.9, 0.3], vec![0.5, 0.5], vec![0.6, 0.4]];
    let ys = vec![0, 0, 1, 1, 2, 2];
    let probe = train_probe(&feats, &ys, 3, 3000, 1.0);
    let acc = probe_accuracy(&probe, &feats, &ys);
    assert!(close(acc, 1.0, 1e-6), "a probe trained long enough fits these 6 points: accuracy {acc}");
    let wrong = Layer { w: vec![vec![0.0, 0.0]; 3], b: vec![0.0, 1.0, 0.0] };
    assert!(close(probe_accuracy(&wrong, &feats, &ys), 1.0 / 3.0, 1e-6), "a probe that always says class 1 is right for 2 of 6");
}

#[test]
fn c4_trained_weights() {
    let net = new_net(&[100, 32, 16, 3], 1);
    for (frozen, expected) in [(2, 51), (1, 579), (0, 3811)] {
        let got = trained_weights(&net, frozen);
        assert_eq!(got, expected, "100 → 32 → 16 → 3 with {frozen} frozen layers: got {got}, expected {expected}");
    }
}
