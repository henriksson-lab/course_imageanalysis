// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day37 object_eval::

use imgcourse::Image;
use day37::object_eval::*;

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-4
}

/// A label image with one square of side `side` at column `x0` (label 1).
fn square(x0: usize, side: usize) -> Image<u32> {
    Image::from_fn(10, 10, |x, y| if (x0..x0 + side).contains(&x) && (2..2 + side).contains(&y) { 1 } else { 0 })
}

/// The IoU table of B2.
fn b2() -> Vec<Vec<f32>> {
    vec![vec![0.82, 0.05, 0.0, 0.0], vec![0.10, 0.45, 0.38, 0.0], vec![0.0, 0.0, 0.0, 0.62]]
}

#[test]
fn c1_iou_of_squares_b1() {
    let t = iou_table(&square(2, 4), &square(3, 4));
    assert!(t.len() == 1 && t[0].len() == 1, "one true and one predicted object should give a 1 × 1 table, got {t:?}");
    assert!(close(t[0][0], 0.6), "B1a, 4 × 4 squares shifted by 1: got {}, expected 0.6", t[0][0]);
    let t = iou_table(&square(2, 2), &square(3, 2));
    assert!(close(t[0][0], 1.0 / 3.0), "B1b, 2 × 2 squares shifted by 1: got {}, expected 1/3", t[0][0]);
}

#[test]
fn c1_iou_table_shape() {
    // Truth: objects 1 (left) and 2 (right); prediction: one object over both, and a speck.
    let truth = Image::from_rows(&[[1, 1, 0, 2, 2], [1, 1, 0, 2, 2], [0, 0, 0, 0, 0]]);
    let pred = Image::from_rows(&[[1, 1, 1, 1, 0], [1, 1, 1, 1, 0], [0, 0, 0, 0, 2]]);
    let t = iou_table(&truth, &pred);
    assert!(t.len() == 2 && t[0].len() == 2, "2 true and 2 predicted objects: expected a 2 × 2 table, got {t:?}");
    assert!(close(t[0][0], 0.5) && close(t[1][0], 2.0 / 10.0) && t[0][1] == 0.0, "a merge: got {t:?}, expected [[0.5, 0], [0.2, 0]]");
}

#[test]
fn c2_match_b2() {
    for (tau, expected) in [(0.5, vec![(0, 0), (2, 3)]), (0.3, vec![(0, 0), (1, 1), (2, 3)]), (0.7, vec![(0, 0)])] {
        let mut m = match_greedy(&b2(), tau);
        m.sort();
        assert_eq!(m, expected, "B2 at tau = {tau}: matched pairs (row, column) from 0");
    }
}

#[test]
fn c2_greedy_b3() {
    let t = vec![vec![0.60, 0.40], vec![0.35, 0.0]];
    let m = match_greedy(&t, 0.3);
    assert_eq!(m, vec![(0, 0)], "B3a: greedy takes G1-P1 and then has nothing left");
}

#[test]
fn c3_scores_b2() {
    for (tp, p, r, f1) in [(2, 0.5, 2.0 / 3.0, 4.0 / 7.0), (3, 0.75, 1.0, 6.0 / 7.0), (1, 0.25, 1.0 / 3.0, 2.0 / 7.0)] {
        let s = scores(3, 4, tp);
        assert!(close(s.precision, p) && close(s.recall, r) && close(s.f1, f1), "B2 with TP = {tp}: got {s:?}, expected precision {p}, recall {r}, F1 {f1}");
    }
    let s = scores(4, 5, 3);
    assert!(close(s.accuracy, 0.5) && close(s.f1, 2.0 / 3.0), "B4c: TP 3, FP 2, FN 1: got {s:?}, expected accuracy 0.5, F1 0.667");
}

#[test]
fn c3_f1_never_rises_with_tau() {
    let mut last = 1.0;
    for k in 0..=20 {
        let tau = k as f32 * 0.05;
        let f1 = scores(3, 4, match_greedy(&b2(), tau).len()).f1;
        assert!(f1 <= last + 1e-6, "B2d: F1 rose from {last} to {f1} at tau = {tau}");
        last = f1;
    }
}

#[test]
fn c4_average_precision_b4() {
    let list = [true, true, false, true, false];
    let ap = average_precision(&list, 4);
    assert!(close(ap, 0.6875), "B4b: got {ap}, expected 0.6875");
    let ap6 = average_precision(&[true, true, false, true, false, false], 4);
    assert!(close(ap6, 0.6875), "B4d: one more FP at the end should not change AP: got {ap6}");
    assert!(close(average_precision(&[false, true], 1), 0.5), "an FP then a TP with one true object: expected AP 0.5");
}
