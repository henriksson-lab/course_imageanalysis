// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p kmeans

use imgcourse::Rng;
use kmeans::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// Numbers on a line as 1D points.
fn line(xs: &[f32]) -> Vec<Vec<f32>> {
    xs.iter().map(|&x| vec![x]).collect()
}

#[test]
fn c1_assign_b1() {
    let points = line(&[1.0, 2.0, 4.0, 7.0, 9.0, 10.0]);
    let (groups, j) = assign(&points, &line(&[1.0, 2.0]));
    assert_eq!(groups, vec![0, 1, 1, 1, 1, 1], "B1, first assign step: groups {{1}}, {{2, 4, 7, 9, 10}}");
    assert!(close(j, 142.0, 1e-3), "B1, first assign step: J = {j}, expected 142");
}

#[test]
fn c2_update_b1() {
    let points = line(&[1.0, 2.0, 4.0, 7.0, 9.0, 10.0]);
    let m = update(&points, &[0, 1, 1, 1, 1, 1], &line(&[1.0, 2.0]));
    assert!(close(m[0][0], 1.0, 1e-5) && close(m[1][0], 6.4, 1e-5), "B1, first update: got {m:?}, expected (1, 6.4)");
    // A centre without points stays.
    let m = update(&points, &[0; 6], &line(&[1.0, 100.0]));
    assert!(close(m[1][0], 100.0, 1e-5), "an empty group's centre should stay at 100, got {}", m[1][0]);
}

#[test]
fn c3_kmeans_b1() {
    let points = line(&[1.0, 2.0, 4.0, 7.0, 9.0, 10.0]);
    let (m, groups, trace) = kmeans(&points, &line(&[1.0, 2.0]), 100);
    let expected = [142.0, 26.84, 15.5, 9.333];
    assert_eq!(trace.len(), 4, "B1: J after each assign step should be {expected:?}, got {trace:?}");
    for k in 0..4 {
        assert!(close(trace[k], expected[k], 0.01), "B1: J after each assign step: got {trace:?}, expected {expected:?}");
    }
    assert!(close(m[0][0], 2.333, 1e-3) && close(m[1][0], 8.667, 1e-3), "B1: final centres {m:?}, expected (2.33, 8.67)");
    assert_eq!(groups, vec![0, 0, 0, 1, 1, 1]);
    // B1(c): from (4, 7) the same answer.
    let (m, _, _) = kmeans(&points, &line(&[4.0, 7.0]), 100);
    assert!(close(m[0][0], 2.333, 1e-3) && close(m[1][0], 8.667, 1e-3), "B1(c): final centres {m:?}, expected (2.33, 8.67)");
}

#[test]
fn c3_kmeans_stuck_b2() {
    let points = line(&[0.0, 1.0, 5.0, 6.0, 10.0, 11.0]);
    let (m, _, trace) = kmeans(&points, &line(&[0.0, 1.0, 8.0]), 100);
    let j = *trace.last().unwrap();
    assert!(close(j, 26.0, 1e-3), "B2(a): J = {j}, expected 26 (stuck)");
    assert!(close(m[2][0], 8.0, 1e-5), "B2(a): the third centre should stay at 8, got {}", m[2][0]);
    let (_, _, trace) = kmeans(&points, &line(&[0.0, 5.0, 10.0]), 100);
    assert!(close(*trace.last().unwrap(), 1.5, 1e-3), "B2(b): the pairs give J = 1.5, got {trace:?}");
}

#[test]
fn c3_kmeans_2d_b4() {
    let points: Vec<Vec<f32>> = [[0.0, 0.0], [0.0, 2.0], [1.0, 1.0], [6.0, 0.0], [6.0, 2.0], [7.0, 1.0]].iter().map(|p| p.to_vec()).collect();
    let (m, _, trace) = kmeans(&points, &[vec![0.0, 0.0], vec![1.0, 1.0]], 100);
    let expected = [90.0, 25.12, 5.333];
    assert_eq!(trace.len(), 3, "B4: J after each assign step should be {expected:?}, got {trace:?}");
    for k in 0..3 {
        assert!(close(trace[k], expected[k], 0.01), "B4: got {trace:?}, expected {expected:?}");
    }
    assert!(close(m[0][0], 0.333, 1e-3) && close(m[1][0], 6.333, 1e-3) && close(m[0][1], 1.0, 1e-4), "B4: centres {m:?}, expected (0.33, 1) and (6.33, 1)");
}

#[test]
fn c4_kmeanspp_probabilities_b3() {
    let points = line(&[0.0, 1.0, 5.0, 6.0, 10.0, 11.0]);
    let p = kmeanspp_probabilities(&points, &line(&[0.0]));
    let expected = [0.0, 1.0 / 283.0, 25.0 / 283.0, 36.0 / 283.0, 100.0 / 283.0, 121.0 / 283.0];
    for i in 0..6 {
        assert!(close(p[i], expected[i], 1e-4), "B3(a): got {p:?}, expected {expected:?}");
    }
    let p = kmeanspp_probabilities(&points, &line(&[0.0, 10.0]));
    assert!(close(p[2] + p[3], 41.0 / 43.0, 1e-4), "B3(b): P(5 or 6) = {}, expected 41/43 = 0.953", p[2] + p[3]);
}

#[test]
fn c4_kmeanspp_restarts() {
    // 100 starts on the points of B2: with k-means++ nearly every run finds J = 1.5.
    let points = line(&[0.0, 1.0, 5.0, 6.0, 10.0, 11.0]);
    let mut rng = Rng::new(1);
    let mut good = 0;
    for _ in 0..100 {
        let start = kmeanspp(&points, 3, &mut rng);
        assert_eq!(start.len(), 3, "k-means++ should give 3 centres");
        let (_, _, trace) = kmeans(&points, &start, 100);
        if close(*trace.last().unwrap(), 1.5, 1e-3) {
            good += 1;
        }
    }
    assert!(good >= 85, "k-means++ starts: {good} of 100 runs reached J = 1.5, expected about 95");
}
