// Checks for the coding exercises, using the numbers of the pen-and-paper exercises (A2 to A4).
// Run: cargo test -p day24 feature_alignment::

use day24::feature_alignment::*;
use imgcourse::Rng;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// The three pairs of A2.
const A2: [Pair; 3] = [((0.0, 0.0), (3.0, 1.0)), ((4.0, 0.0), (7.0, 1.0)), ((0.0, 4.0), (3.0, 6.0))];

#[test]
fn c1_fit_translation_a2() {
    let (tx, ty) = fit_translation(&A2);
    assert!(close(tx, 3.0, 1e-5) && close(ty, 4.0 / 3.0, 1e-5), "A2(a): got ({tx}, {ty}), expected (3, 4/3)");
}

#[test]
fn c2_similarity_from_two_a2() {
    let m = similarity_from_two(A2[0].0, A2[0].1, A2[1].0, A2[1].1);
    let (x, y) = apply(&m, (0.0, 4.0));
    assert!(close(x, 3.0, 1e-4) && close(y, 5.0, 1e-4), "A2(b): (0, 4) is mapped to ({x}, {y}), expected (3, 5)");
    // Turned by 90° and scaled by 2: (0, 0) → (1, 1) and (1, 0) → (1, 3).
    let m = similarity_from_two((0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (1.0, 3.0));
    let (x, y) = apply(&m, (0.0, 1.0));
    assert!(close(x, -1.0, 1e-4) && close(y, 1.0, 1e-4), "turn by 90°, scale 2: (0, 1) → ({x}, {y}), expected (−1, 1)");
}

#[test]
fn c3_fit_affine_exact_and_least_squares() {
    // A4's transform, A = [[1, 1], [0, 2]], t = (3, −1): three pairs fix it exactly.
    let truth: Affine = [[1.0, 1.0, 3.0], [0.0, 2.0, -1.0]];
    let pts = [(1.0, 1.0), (2.0, 0.0), (0.0, 3.0)];
    let pairs: Vec<Pair> = pts.iter().map(|&p| (p, apply(&truth, p))).collect();
    let m = fit_affine(&pairs).expect("three points not on a line must give a transform");
    for r in 0..2 {
        for c in 0..3 {
            assert!(close(m[r][c], truth[r][c], 1e-4), "exact fit: got {m:?}, expected {truth:?}");
        }
    }
    // More pairs with errors of ±0.5 that cancel: least squares averages them away.
    let mut more = Vec::new();
    for (k, p) in [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)].iter().enumerate() {
        let (u, v) = apply(&truth, *p);
        let e = if k == 0 || k == 3 { 0.5 } else { -0.5 };
        more.push((*p, (u + e, v)));
    }
    let m = fit_affine(&more).unwrap();
    assert!(close(m[0][0], 1.0, 1e-3) && close(m[0][1], 1.0, 1e-3), "least squares: got {m:?}");
    // Points on a line give no transform.
    let line = [((0.0, 0.0), (1.0, 1.0)), ((1.0, 1.0), (2.0, 2.0)), ((2.0, 2.0), (3.0, 3.0))];
    assert!(fit_affine(&line).is_none(), "three points on a line: expected None");
}

#[test]
fn c4_inliers_a4() {
    let m: Affine = [[1.0, 1.0, 3.0], [0.0, 2.0, -1.0]];
    let pairs = [((1.0, 1.0), (5.0, 1.0)), ((2.0, 0.0), (6.0, 0.0)), ((0.0, 3.0), (4.0, 6.0)), ((4.0, 4.0), (15.0, 3.0))];
    let got = inliers(&m, &pairs, 2.0);
    assert_eq!(got, vec![0, 1], "A4: the first two are inliers (distances 0 and 1.41), the others not (2.24, 5.66)");
}

#[test]
fn c4_ransac_iterations_a3() {
    let row: Vec<usize> = (1..=4).map(|s| ransac_iterations(0.7, s, 0.99)).collect();
    assert_eq!(row, vec![4, 7, 11, 17], "A3(a), w = 0.7, s = 1..4: got {row:?}, expected [4, 7, 11, 17]");
    assert_eq!(ransac_iterations(0.3, 4, 0.99), 567, "w = 0.3, homography: 567");
    assert_eq!(ransac_iterations(0.5, 4, 0.99), 72, "w = 0.5, homography: 72");
}

#[test]
fn c5_ransac_affine_finds_the_inliers() {
    let truth: Affine = [[0.9, -0.3, 12.0], [0.25, 1.1, -5.0]];
    let mut rng = Rng::new(5);
    let mut pairs = Vec::new();
    for k in 0..60 {
        let p = (100.0 * rng.uniform(), 100.0 * rng.uniform());
        // 25 correct pairs, 35 wrong ones (w = 0.42).
        let q = if k < 25 { apply(&truth, p) } else { (100.0 * rng.uniform(), 100.0 * rng.uniform()) };
        pairs.push((p, q));
    }
    let (m, found) = ransac_affine(&pairs, 1.0, 200, &mut Rng::new(1)).expect("RANSAC found nothing");
    assert_eq!(found, (0..25).collect::<Vec<_>>(), "the 25 correct pairs are the inliers, got {found:?}");
    for r in 0..2 {
        for c in 0..3 {
            assert!(close(m[r][c], truth[r][c], 1e-3), "RANSAC transform: got {m:?}, expected {truth:?}");
        }
    }
}
