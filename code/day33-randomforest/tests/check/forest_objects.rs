// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day33 forest_objects::
//
// In B1, features are 0 = area and 1 = texture; classes 0 = interphase, 1 = mitotic.

use day33::forest_objects::*;
use imgcourse::Rng;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn leaf(i: f32, m: f32) -> Box<Tree> {
    Box::new(Tree::Leaf { counts: vec![i, m] })
}

/// The depth-2 tree of B1.
fn b1_tree() -> Tree {
    Tree::Split {
        feature: 1,
        threshold: 0.0,
        counts: vec![50.0, 50.0],
        left: Box::new(Tree::Split { feature: 0, threshold: 0.0, counts: vec![45.0, 10.0], left: leaf(40.0, 2.0), right: leaf(5.0, 8.0) }),
        right: Box::new(Tree::Split { feature: 1, threshold: 0.0, counts: vec![5.0, 40.0], left: leaf(5.0, 5.0), right: leaf(0.0, 35.0) }),
    }
}

#[test]
fn c1_gini_b1() {
    for (counts, expected) in [([50., 50.], 0.5), ([45., 10.], 0.298), ([5., 40.], 0.198), ([40., 2.], 0.091), ([5., 8.], 0.473), ([0., 35.], 0.0)] {
        let g = gini(&counts);
        assert!(close(g, expected, 1e-3), "B1 (a): node {counts:?}: got {g:.3}, expected {expected}");
    }
}

#[test]
fn c2_impurity_importance_b1() {
    let imp = impurity_importance(&[b1_tree()], 2);
    assert!(close(imp[1], 0.286, 2e-3), "B1 (c): texture: got {:.3}, expected 0.247 + 0.039 = 0.286", imp[1]);
    assert!(close(imp[0], 0.064, 2e-3), "B1 (c): area: got {:.3}, expected 0.064", imp[0]);
    let share = imp[1] / (imp[0] + imp[1]);
    assert!(close(share, 0.82, 0.01), "B1 (c): texture's share: got {share:.2}, expected 0.82");
    // Two copies of the tree: the average is the same.
    let imp2 = impurity_importance(&[b1_tree(), b1_tree()], 2);
    assert!(close(imp2[1], imp[1], 1e-6), "averaged over two identical trees: got {}, expected {}", imp2[1], imp[1]);
}

#[test]
fn c2_adds_up_b6() {
    // B6: in one tree the importances add up to the root impurity minus the weighted leaf
    // impurities; for a tree grown until its leaves are pure, to the root impurity.
    let imp = impurity_importance(&[b1_tree()], 2);
    let leaves = (42.0 * gini(&[40., 2.]) + 13.0 * gini(&[5., 8.]) + 10.0 * gini(&[5., 5.])) / 100.0;
    assert!(close(imp[0] + imp[1], 0.5 - leaves, 1e-5), "B6: sum {} should be 0.5 − {leaves}", imp[0] + imp[1]);
}

/// 300 objects: feature 0 decides the class (with some overlap), feature 1 is noise, feature
/// 2 is an exact copy of feature 0.
fn table() -> (Vec<Vec<f32>>, Vec<usize>) {
    let mut rng = Rng::new(4);
    let (mut points, mut labels) = (Vec::new(), Vec::new());
    for i in 0..300 {
        let class = i % 2;
        let f0 = class as f32 + 0.4 * rng.normal();
        points.push(vec![f0, rng.uniform(), f0]);
        labels.push(class);
    }
    (points, labels)
}

#[test]
fn c3_permutation_b3() {
    let (points, labels) = table();
    let settings = Settings { n_trees: 40, m: 1, class_weights: vec![1.0, 1.0], seed: 1 };
    let forest = Forest::train(&points, &labels, 2, &settings);
    let mut rng = Rng::new(9);
    let noise = permutation_importance(&forest, &points, &labels, &[1], 3, &mut rng);
    let alone = permutation_importance(&forest, &points, &labels, &[0], 3, &mut rng);
    let group = permutation_importance(&forest, &points, &labels, &[0, 2], 3, &mut rng);
    assert!(noise.abs() < 0.05, "the noise column: got {noise:.3}, expected about 0");
    assert!(group > 0.25, "feature 0 and its copy shuffled together: got {group:.3}, expected a large rise (> 0.25)");
    assert!(alone < group / 2.0, "B3 (c): feature 0 alone ({alone:.3}) should look much less important than the group ({group:.3}): its copy still carries it");
}

#[test]
fn c4_correlation() {
    let a: Vec<f32> = (1..=20).map(|i| i as f32).collect();
    let up: Vec<f32> = a.iter().map(|v| 2.0 * v + 1.0).collect();
    let down: Vec<f32> = a.iter().map(|v| 5.0 - v).collect();
    assert!(close(correlation(&a, &up), 1.0, 1e-5), "a line going up: got {}, expected 1", correlation(&a, &up));
    assert!(close(correlation(&a, &down), -1.0, 1e-5), "a line going down: got {}, expected −1", correlation(&a, &down));
    // B3: area and equivalent diameter d = 2 √(A / π) are almost perfectly correlated.
    let area: Vec<f32> = (0..50).map(|i| 200.0 + 4.0 * i as f32).collect();
    let diameter: Vec<f32> = area.iter().map(|a| 2.0 * (a / std::f32::consts::PI).sqrt()).collect();
    let r = correlation(&area, &diameter);
    assert!(r > 0.99, "B3: area and diameter: got {r:.4}, expected above 0.99");
}
