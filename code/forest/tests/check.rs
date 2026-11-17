// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p forest
//
// Classes: 0 = background (B), 1 = nucleus (N).

use forest::*;
use imgcourse::Rng;

const B: usize = 0;
const N: usize = 1;

fn mean_oob_fraction(n: usize, samples: usize) -> f32 {
    let mut rng = Rng::new(1);
    let mut total = 0.0;
    for _ in 0..samples {
        let s = bootstrap(n, &mut rng);
        assert_eq!(s.len(), n, "a bootstrap sample has n = {n} draws, got {}", s.len());
        assert!(s.iter().all(|&i| i < n), "draws must be point numbers 0 .. {}", n - 1);
        let bag = in_bag(&s, n);
        total += bag.iter().filter(|&&b| !b).count() as f32 / n as f32;
    }
    total / samples as f32
}

#[test]
fn c1_one_in_three_b1() {
    let f5 = mean_oob_fraction(5, 4000);
    assert!((f5 - 0.328).abs() < 0.015, "B1 (a): n = 5, mean out-of-bag fraction: got {f5:.3}, expected (1 − 1/5)^5 = 0.328");
    let f20 = mean_oob_fraction(20, 2000);
    assert!((f20 - 0.358).abs() < 0.01, "n = 20: got {f20:.3}, expected (1 − 1/20)^20 = 0.358");
}

#[test]
fn c1_in_bag() {
    assert_eq!(in_bag(&[0, 2, 2], 4), vec![true, false, true, false], "sample [0, 2, 2] of 4 points");
}

#[test]
fn c2_features_b4() {
    let mut rng = Rng::new(2);
    let mut texture = 0;
    for _ in 0..4000 {
        let f = choose_features(10, 3, &mut rng);
        assert_eq!(f.len(), 3, "m = 3 features, got {f:?}");
        assert!(f.windows(2).all(|w| w[0] < w[1]), "all different and in increasing order: got {f:?}");
        assert!(f.iter().all(|&j| j < 10), "features 0 .. 9: got {f:?}");
        texture += f.contains(&0) as usize;
    }
    let share = texture as f32 / 4000.0;
    assert!((share - 0.3).abs() < 0.03, "B4 (b): feature 0 should be a candidate in 3/10 of the draws, got {share:.3}");
    assert_eq!(choose_features(4, 4, &mut rng), vec![0, 1, 2, 3], "m = d: every feature (bagging)");
}

#[test]
fn c3_train() {
    // Two clusters, separable along feature 0; feature 1 is noise.
    let mut rng = Rng::new(3);
    let (mut points, mut labels) = (Vec::new(), Vec::new());
    for i in 0..60 {
        let class = i % 2;
        points.push(vec![class as f32 * 2.0 + rng.uniform(), rng.uniform()]);
        labels.push(class);
    }
    let forest = train_forest(&points, &labels, 2, 25, 1, &mut rng);
    assert_eq!(forest.trees.len(), 25, "25 trees");
    assert_eq!(forest.in_bag.len(), 25, "one in-bag list per tree");
    let share: f32 = forest.in_bag.iter().flatten().filter(|&&b| b).count() as f32 / (25.0 * 60.0);
    assert!((share - 0.632).abs() < 0.05, "about 63 % of the points should be in each bootstrap sample, got {share:.3}");
    assert_eq!(error_rate(&forest.trees, &points, &labels, 2), 0.0, "separable data: no training errors");
    assert!(oob_error(&forest, &points, &labels, 2) < 0.05, "separable data: OOB error near 0");
}

/// The five trees of B3, each voting for one class everywhere: N, N, B, N, B.
fn b3_trees() -> Vec<Tree> {
    [N, N, B, N, B].iter().map(|&c| Tree::vote_for(c, 2)).collect()
}

#[test]
fn c4_vote_b3() {
    let p = vote(&b3_trees(), &[0.0], 2);
    assert!((p[N] - 0.6).abs() < 1e-6, "B3 (a): P(N): got {}, expected 0.6", p[N]);
    assert_eq!(label(&p), N, "B3 (a): label N");
}

#[test]
fn c5_oob_b3() {
    // Trees 2, 3 and 5 did not see the pixel: their votes N, B, B give the OOB label B.
    let forest = Forest { trees: b3_trees(), in_bag: vec![vec![true], vec![false], vec![false], vec![true], vec![false]] };
    let e = oob_error(&forest, &[vec![0.0]], &[B], 2);
    assert_eq!(e, 0.0, "B3 (b): the pixel is truly B and the OOB label is B: no OOB error, got {e}");
    let e = oob_error(&forest, &[vec![0.0]], &[N], 2);
    assert_eq!(e, 1.0, "if the pixel were truly N, the OOB label B would be an error, got {e}");
}
