// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day40 yolo::

use imgcourse::Image;
use day40::yolo::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn det(x0: f32, y0: f32, x1: f32, y1: f32, score: f32, class: usize) -> Detection {
    Detection { x0, y0, x1, y1, score, class }
}

#[test]
fn c1_target_b7() {
    let p = distance_target(2.3);
    assert!(close(p[2], 0.7, 1e-5) && close(p[3], 0.3, 1e-5), "y = 2.3: got {:.2} on bin 2 and {:.2} on bin 3, expected 0.7 and 0.3 (B7)", p[2], p[3]);
    assert!(close(p.iter().sum::<f32>(), 1.0, 1e-5), "y = 2.3: the probabilities sum to {}, expected 1", p.iter().sum::<f32>());
    for y in [0.0, 4.0, 7.75, 14.5, 15.0, 20.0] {
        let p = distance_target(y);
        let mean: f32 = p.iter().enumerate().map(|(k, v)| k as f32 * v).sum();
        let want = y.min(15.0);
        assert!(close(mean, want, 1e-4), "y = {y}: the mean of the target is {mean}, expected {want}");
    }
}

#[test]
fn c2_decode_b1() {
    let mut dists = [[0.0; BINS]; 4];
    dists[0][2] = 0.5; // left: half on 2, half on 3
    dists[0][3] = 0.5;
    dists[1][1] = 1.0; // top: all on 1
    dists[2][1] = 0.5; // right: half on 1, half on 2
    dists[2][2] = 0.5;
    dists[3][2] = 1.0; // bottom: all on 2
    let b = decode_cell(16.0, 5, 7, &dists, 0.77, 0);
    let got = [b.x0, b.y0, b.x1, b.y1];
    let expected = [80.0, 72.0, 144.0, 120.0];
    for i in 0..4 {
        assert!(close(got[i], expected[i], 1e-3), "B1c: got the box {got:?}, expected (80, 72)–(144, 120)");
    }
    let truth = det(84.0, 70.0, 150.0, 122.0, 1.0, 0);
    let v = iou(&b, &truth);
    assert!(close(v, 0.79, 0.005), "B1d: IoU with the true box: got {v}, expected 0.79");
    // B1f: right 0.6 on 2 and 0.4 on 9
    dists[2] = [0.0; BINS];
    dists[2][2] = 0.6;
    dists[2][9] = 0.4;
    let b = decode_cell(16.0, 5, 7, &dists, 0.77, 0);
    assert!(close(b.x1, 196.8, 1e-3), "B1f: right edge got {}, expected 120 + 4.8 · 16 = 196.8", b.x1);
}

/// The six boxes of B3; class 0 = cell, 1 = debris.
fn b3() -> Vec<Detection> {
    vec![
        det(10.0, 10.0, 50.0, 50.0, 0.90, 0),
        det(14.0, 12.0, 54.0, 52.0, 0.80, 0),
        det(40.0, 10.0, 80.0, 50.0, 0.75, 0),
        det(12.0, 10.0, 52.0, 50.0, 0.60, 1),
        det(44.0, 14.0, 84.0, 54.0, 0.50, 0),
        det(100.0, 100.0, 120.0, 120.0, 0.20, 0),
    ]
}

fn names(kept: &[usize]) -> Vec<String> {
    kept.iter().map(|i| format!("b{}", i + 1)).collect()
}

#[test]
fn c3_class_aware_and_agnostic_b3() {
    let boxes = b3();
    for (t, aware, expected) in [(0.5, true, vec![0, 2, 3]), (0.5, false, vec![0, 2]), (0.2, true, vec![0, 2, 3]), (0.2, false, vec![0, 2])] {
        let kept = nms_classes(&boxes, 0.25, t, aware);
        let mut sorted = kept.clone();
        sorted.sort();
        let kind = if aware { "class-aware" } else { "class-agnostic" };
        assert_eq!(sorted, expected, "{kind}, NMS {t}: kept {:?}, expected {:?} (B3)", names(&kept), names(&expected));
    }
}

#[test]
fn c3_not_nested_b10() {
    // c debris, a cell, b cell (indices 0, 1, 2)
    let boxes = vec![det(0.0, 0.0, 40.0, 40.0, 0.9, 1), det(10.0, 0.0, 50.0, 40.0, 0.8, 0), det(20.0, 0.0, 60.0, 40.0, 0.7, 0)];
    let n = ["c", "a", "b"];
    let aware = nms_classes(&boxes, 0.0, 0.5, true);
    let agnostic = nms_classes(&boxes, 0.0, 0.5, false);
    let show = |k: &[usize]| k.iter().map(|&i| n[i]).collect::<Vec<_>>();
    assert_eq!(aware, vec![0, 1], "class-aware: kept {:?}, expected [c, a] (B10)", show(&aware));
    assert_eq!(agnostic, vec![0, 2], "class-agnostic: kept {:?}, expected [c, b] (B10)", show(&agnostic));
}

#[test]
fn c4_peaks_b5() {
    let map = Image::from_rows(&[[1., 2., 1., 0., 0.], [2., 5., 2., 0., 3.], [1., 2., 1., 0., 0.], [0., 0., 0., 4., 1.], [3., 0., 2., 0., 2.]]);
    // the exercise counts rows and columns from 1; (x, y) = (column − 1, row − 1)
    let p = heatmap_peaks(&map, 1, 0.0);
    assert_eq!(p, vec![(1, 1), (4, 1), (3, 3), (0, 4)], "3 × 3 window: got {p:?}, expected 5, 3, 4 and 3 at (1, 1), (4, 1), (3, 3), (0, 4) (B5a)");
    let p = heatmap_peaks(&map, 1, 3.5);
    assert_eq!(p, vec![(1, 1), (3, 3)], "threshold 3.5: got {p:?}, expected the 5 and the 4 (B5b)");
    let p = heatmap_peaks(&map, 2, 0.0);
    assert_eq!(p, vec![(1, 1), (0, 4)], "5 × 5 window: got {p:?}, expected the 5 and the 3 in the bottom-left corner (B5b)");
}
