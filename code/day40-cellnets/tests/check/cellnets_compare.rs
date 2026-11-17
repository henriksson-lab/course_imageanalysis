// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day40 cellnets_compare::

use day40::cellnets_compare::*;
use imgcourse::Image;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// The IoU table of B2: rows T1–T4, columns P1–P5.
fn b2_iou() -> Vec<Vec<f32>> {
    vec![
        vec![0.0, 0.0, 0.40, 0.0, 0.0],
        vec![0.0, 0.0, 0.47, 0.07, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.75],
        vec![0.0, 0.0, 0.0, 0.78, 0.0],
    ]
}

#[test]
fn c1_f1_b2() {
    let f1 = f1_curve(&b2_iou(), 5, &[0.5, 0.7, 0.8]);
    let expected = [4.0 / 9.0, 4.0 / 9.0, 0.0];
    for i in 0..3 {
        assert!(close(f1[i], expected[i], 1e-5), "F1 at τ = 0.5, 0.7, 0.8: got {f1:?}, expected 0.44, 0.44, 0 (B2a)");
    }
}

#[test]
fn c1_f1_b1() {
    // B1, method A: 20 true, 21 predicted, 18 pairs above 0.5 of which 12 above 0.7
    let mut iou = vec![vec![0.0; 21]; 20];
    for i in 0..18 {
        iou[i][i] = if i < 12 { 0.8 } else { 0.6 };
    }
    let f1 = f1_curve(&iou, 21, &[0.5, 0.7]);
    assert!(close(f1[0], 36.0 / 41.0, 1e-5) && close(f1[1], 24.0 / 41.0, 1e-5), "method A: got {f1:?}, expected 0.88 and 0.59 (B1a)");
}

/// The overlap table of B2, rebuilt from its IoUs and areas: row 0 and column 0 are the background.
fn b2_overlaps() -> Vec<Vec<usize>> {
    vec![
        vec![0, 47, 69, 55, 39, 0],    // background: P1, P2 and the rims of P3, P4
        vec![23, 0, 0, 322, 0, 0],     // T1 (345 px): 322 inside P3
        vec![13, 0, 0, 406, 69, 0],    // T2 (488 px): 406 inside P3, 69 inside P4
        vec![166, 0, 0, 0, 0, 492],    // T3 (658 px)
        vec![21, 0, 0, 0, 457, 0],     // T4 (478 px)
    ]
}

#[test]
fn c2_merge_b2() {
    let (m, s) = merges_and_splits(&b2_overlaps());
    assert_eq!((m, s), (1, 0), "B2b: got {m} merges and {s} splits, expected 1 merge (P3 covers 0.93 of T1 and 0.83 of T2) and no split");
}

#[test]
fn c2_split_and_merge_images() {
    // truth: two squares side by side; prediction 1 covers both (a merge)
    let truth = Image::from_fn(12, 4, |x, _| if x < 6 { 1 } else { 2 });
    let merged = Image::new(12, 4, 1u32);
    assert_eq!(merges_and_splits(&overlaps(&truth, &merged)), (1, 0), "one prediction over two true objects should be 1 merge, 0 splits");
    // truth: one rectangle; prediction: its two halves (a split)
    let one = Image::new(12, 4, 1u32);
    let halves = Image::from_fn(12, 4, |x, _| if x < 6 { 1 } else { 2 });
    assert_eq!(merges_and_splits(&overlaps(&one, &halves)), (0, 1), "two predictions inside one true object should be 0 merges, 1 split");
}

#[test]
fn c3_bootstrap_b6() {
    let d = [0.042, 0.021, 0.431, 0.053, -0.015, 0.0, 0.0];
    let (mean, lo, hi) = bootstrap_interval(&d, 10_000, 1);
    assert!(close(mean, 0.076, 0.001), "mean difference: got {mean}, expected 0.076 (B6a)");
    assert!(lo > -0.02 && lo < 0.03 && hi > 0.15 && hi < 0.25, "95 % interval: got [{lo:.3}, {hi:.3}], expected about [0.004, 0.199] (B6c)");
    let (m, l, h) = bootstrap_interval(&[0.1; 5], 1000, 2);
    assert!(close(m, 0.1, 1e-6) && close(l, 0.1, 1e-6) && close(h, 0.1, 1e-6), "equal differences: got ({m}, {l}, {h}), expected the interval to be the single value 0.1");
}
