// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day12 seg_eval::

use imgcourse::Image;
use day12::seg_eval::*;

/// The image of B1 with grey levels 0–90. Level `i` is stored as `i / 128`, so that with
/// 128 bins it lands in bin `i` and "object where the bin is above 45" means I > 45.
fn b1_image() -> Image<f32> {
    let levels = [
        [10., 20., 30., 20., 10.],
        [20., 70., 80., 60., 20.],
        [30., 80., 90., 40., 50.],
        [20., 60., 40., 30., 20.],
        [10., 20., 30., 70., 10.],
    ];
    Image::from_rows(&levels).map(|v| v / 128.0)
}

fn b1_truth() -> Image<bool> {
    let g = [[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]];
    Image::from_rows(&g).map(|v| v == 1)
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-3
}

#[test]
fn c1_confusion_b1() {
    let pred = b1_image().map(|v| v * 128.0 > 45.0);
    let c = confusion(&pred, &b1_truth());
    assert_eq!(c, Counts { tp: 6, fp: 2, fn_: 1, tn: 16 }, "B1 (a): the confusion matrix at t = 45");
}

#[test]
fn c2_measures_b1() {
    let c = Counts { tp: 6, fp: 2, fn_: 1, tn: 16 };
    let got = [c.precision(), c.recall(), c.fpr(), c.accuracy(), c.dice(), c.jaccard()];
    let expected = [0.750, 0.857, 0.111, 0.880, 0.800, 0.667];
    let names = ["precision", "recall", "FPR", "accuracy", "Dice", "Jaccard"];
    for i in 0..6 {
        assert!(close(got[i], expected[i]), "B1 (b): {}: got {:.3}, expected {}", names[i], got[i], expected[i]);
    }
}

#[test]
fn c2_all_background_b3() {
    // B3 (a): 3 % objects, everything called background
    let c = Counts { tp: 0, fp: 0, fn_: 3, tn: 97 };
    assert!(close(c.accuracy(), 0.97) && c.dice() == 0.0, "B3 (a): accuracy {} (expected 0.97), Dice {} (expected 0)", c.accuracy(), c.dice());
}

#[test]
fn c3_roc_points_b2() {
    let points = roc(&b1_image(), &b1_truth(), 128);
    assert_eq!(points.len(), 128, "one point per threshold 0..127");
    for (t, fpr, tpr) in [(35, 3.0 / 18.0, 1.0), (45, 2.0 / 18.0, 6.0 / 7.0), (55, 1.0 / 18.0, 6.0 / 7.0), (75, 0.0, 3.0 / 7.0)] {
        let (f, r) = points[t];
        assert!(close(f, fpr) && close(r, tpr), "B2 (a): t = {t}: got (FPR, TPR) = ({f:.3}, {r:.3}), expected ({fpr:.3}, {tpr:.3})");
    }
}

#[test]
fn c4_auc_b9() {
    let a = auc(&roc(&b1_image(), &b1_truth(), 128));
    assert!(close(a, 121.0 / 126.0), "B9: AUC: got {a:.4}, expected 121/126 = 0.9603");
}

#[test]
fn c4_auc_chance_and_perfect() {
    assert!(close(auc(&[(0.5, 0.5)]), 0.5), "the diagonal: AUC 0.5");
    assert!(close(auc(&[(0.0, 1.0)]), 1.0), "a perfect classifier: AUC 1");
}

#[test]
fn c5_best_dice_b2() {
    let (t, d) = best_dice(&b1_image(), &b1_truth(), 128);
    // Dice is 6/7 = 0.857 for every t from 50 to 59 (B2 (b) checks 55); the smallest wins
    assert!(close(d, 6.0 / 7.0), "B2 (b): best Dice: got {d:.3}, expected 0.857");
    assert_eq!(t, 50, "B2 (b): the smallest threshold with Dice 0.857");
}
