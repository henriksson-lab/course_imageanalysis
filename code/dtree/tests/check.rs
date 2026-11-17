// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p dtree
//
// Classes: 0 = background (B), 1 = nucleus (N). Features: 0 = intensity I, 1 = local variance V.

use dtree::*;

const B: usize = 0;
const N: usize = 1;

/// The eight pixels of B1.
fn b1() -> (Vec<Vec<f32>>, Vec<usize>) {
    let i = [20., 35., 45., 50., 60., 80., 90., 110.];
    let v = [2., 3., 5., 9., 12., 4., 15., 11.];
    let class = [B, B, N, B, N, B, N, N];
    ((0..8).map(|k| vec![i[k], v[k]]).collect(), class.to_vec())
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-3
}

#[test]
fn c1_gini() {
    for (counts, expected, name) in [
        (vec![4, 4], 0.5, "B1 (a): 4 B and 4 N"),
        (vec![2, 4], 4.0 / 9.0, "2 B and 4 N"),
        (vec![3, 0], 0.0, "a pure node"),
        (vec![300, 100], 0.375, "B10: (300, 100)"),
        (vec![1, 1, 1], 2.0 / 3.0, "three classes, equal shares"),
    ] {
        let g = gini(&counts);
        assert!(close(g, expected), "{name}: got {g}, expected {expected}");
    }
}

#[test]
fn c2_best_split_b1() {
    let (points, labels) = b1();
    let s = best_split(&points, &labels, 2, 1).expect("B1 has splits");
    assert_eq!(s.feature, 1, "B1 (c): the root should ask about V (feature 1), got feature {}", s.feature);
    assert!(close(s.threshold, 4.5), "B1 (c): threshold: got {}, expected 4.5", s.threshold);
    assert!(close(s.decrease, 0.3), "B1 (c): decrease: got {}, expected 0.5 − 0.2 = 0.3", s.decrease);
    // intensity alone: I ≤ 40 (the first of two equally good thresholds)
    let only_i: Vec<Vec<f32>> = points.iter().map(|p| vec![p[0]]).collect();
    let s = best_split(&only_i, &labels, 2, 1).unwrap();
    assert!(close(s.threshold, 40.0), "B1 (b): best threshold on I: got {}, expected 40", s.threshold);
    assert!(close(s.decrease, 0.5 - 1.0 / 3.0), "B1 (b): decrease: got {}, expected 0.5 − 0.333", s.decrease);
}

#[test]
fn c2_min_leaf() {
    let (points, labels) = b1();
    // With at least 4 points per child, only V ≤ 7 (or I ≤ 55) is allowed.
    let s = best_split(&points, &labels, 2, 4).unwrap();
    let n_left = points.iter().filter(|p| p[s.feature] <= s.threshold).count();
    assert_eq!(n_left, 4, "min_leaf 4 on 8 points: the left child must have exactly 4 points, has {n_left}");
    assert!(best_split(&points, &labels, 2, 5).is_none(), "min_leaf 5 on 8 points: no split is allowed");
}

#[test]
fn c3_grow_b1() {
    let (points, labels) = b1();
    let stump = grow(&points, &labels, 2, 1, 1);
    match &stump {
        Tree::Split { feature: 1, threshold, left, right } => {
            assert!(close(*threshold, 4.5), "stump threshold: got {threshold}, expected 4.5");
            assert_eq!(**left, Tree::Leaf { counts: vec![3, 0] }, "left leaf of the stump: 3 B");
            assert_eq!(**right, Tree::Leaf { counts: vec![1, 4] }, "right leaf of the stump: 1 B, 4 N");
        }
        other => panic!("depth 1: expected one split on V at 4.5, got {other:?}"),
    }
    let full = grow(&points, &labels, 2, 20, 1);
    assert_eq!(error_rate(&full, &points, &labels), 0.0, "grown fully, the tree should get every training point right");
}

#[test]
fn c3_log_invariance_b5() {
    // B5: store log(I) instead of I: the tree asks the same questions (moved thresholds).
    let (points, labels) = b1();
    let logged: Vec<Vec<f32>> = points.iter().map(|p| vec![p[0].ln(), p[1]]).collect();
    let a = grow(&points, &labels, 2, 20, 1);
    let b = grow(&logged, &labels, 2, 20, 1);
    assert_eq!(leaves(&a), leaves(&b), "the same number of leaves");
    for (p, q) in points.iter().zip(&logged) {
        assert_eq!(predict(&a, p), predict(&b, q), "point {p:?}: the two trees disagree");
    }
}

#[test]
fn c4_predict_b2() {
    let leaf = |class: usize| Box::new(Tree::Leaf { counts: if class == B { vec![1, 0] } else { vec![0, 1] } });
    let tree = Tree::Split {
        feature: 1,
        threshold: 6.0,
        left: Box::new(Tree::Split { feature: 0, threshold: 100.0, left: leaf(B), right: leaf(N) }),
        right: Box::new(Tree::Split { feature: 0, threshold: 40.0, left: leaf(B), right: leaf(N) }),
    };
    for (x, expected) in [([30., 10.], B), ([120., 3.], N), ([70., 5.], B)] {
        let got = predict(&tree, &x);
        assert_eq!(got, expected, "B2 (b): (I, V) = {x:?}: got class {got}, expected {expected} (0 = B, 1 = N)");
    }
}

#[test]
fn c5_regression_b6() {
    let xs = [1., 2., 3., 4., 5., 6.];
    let ys = [10., 12., 11., 30., 32., 31.];
    let (threshold, sse) = regression_stump(&xs, &ys);
    assert!(close(threshold, 3.5), "B6: split between x = 3 and 4, threshold 3.5; got {threshold}");
    assert!(close(sse, 4.0), "B6: sum of squared errors: got {sse}, expected 4");
}
