// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p training training_curves::

use training::training_curves::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn c1_smooth_by_hand() {
    // B6: beta 0.5 on 1, 0, 1, 0: 1, 0.5, 0.75, 0.375
    let s = smooth(&[1.0, 0.0, 1.0, 0.0], 0.5);
    let want = [1.0, 0.5, 0.75, 0.375];
    assert!(s.len() == 4 && (0..4).all(|i| close(s[i], want[i], 1e-6)), "B6: got {s:?}, expected {want:?}");
    let flat = smooth(&[0.3; 10], 0.9);
    assert!(flat.iter().all(|v| close(*v, 0.3, 1e-6)), "a flat curve stays flat");
}

#[test]
fn c2_early_stopping_b5() {
    // B5: lowest at epoch 3; with patience 3 training stops at epoch 6
    let val = [0.9, 0.7, 0.6, 0.55, 0.56, 0.58, 0.57, 0.6, 0.62];
    assert_eq!(early_stopping(&val, 3), (3, 6), "B5: patience 3");
    assert_eq!(early_stopping(&val, 10), (3, 8), "patience longer than the run: stop at the last epoch");
    let falling = [1.0, 0.8, 0.6, 0.4];
    assert_eq!(early_stopping(&falling, 2), (3, 3), "still improving: best and stop are the last epoch");
}

#[test]
fn c3_chance_loss_b2() {
    // B2: ln K for K balanced classes; 90/10 gives 0.325
    assert!(close(chance_loss(&[5, 5, 5]), 3f32.ln(), 1e-5), "three balanced classes: ln 3 = 1.0986");
    assert!(close(chance_loss(&[1, 1]), 2f32.ln(), 1e-5), "two classes: ln 2 = 0.693");
    assert!(close(chance_loss(&[90, 10]), 0.3251, 1e-3), "B2(c): 90 % / 10 %: got {}, expected 0.325", chance_loss(&[90, 10]));
    assert!(close(chance_loss(&[10, 0]), 0.0, 1e-6), "one class only: 0");
}

#[test]
fn c4_class_weights_b3() {
    // B3: 240, 50, 10 -> 300 / (3 n_k) = 0.4167, 2, 10
    let w = class_weights(&[240, 50, 10]);
    let want = [0.41667, 2.0, 10.0];
    assert!((0..3).all(|k| close(w[k], want[k], 1e-4)), "B3: got {w:?}, expected {want:?}");
    let total: f32 = [240.0, 50.0, 10.0].iter().zip(&w).map(|(n, wk)| n * wk).sum();
    assert!(close(total, 300.0, 1e-3), "the weights over all examples add up to the number of examples");
}

#[test]
fn c5_focal_loss_b4() {
    // B4: p = 0.9: cross-entropy 0.105, focal (gamma 2) 0.00105; p = 0.1: 2.303 and 1.865
    assert!(close(focal_loss(&[0.9, 0.1], 0, 0.0), 0.10536, 1e-4), "gamma 0 is the cross-entropy");
    assert!(close(focal_loss(&[0.9, 0.1], 0, 2.0), 0.0010536, 1e-6), "B4: p = 0.9, gamma 2: expected 0.00105");
    assert!(close(focal_loss(&[0.9, 0.1], 1, 2.0), 1.8651, 1e-3), "B4: p = 0.1, gamma 2: expected 1.865");
}
