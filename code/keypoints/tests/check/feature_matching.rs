// Checks for the coding exercises, using the numbers of the pen-and-paper exercises (A1 to A3, B6).
// Run: cargo test -p keypoints feature_matching::

use keypoints::feature_matching::*;

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-4
}

/// The table of A1: rows 1 to 4 (here 0 to 3), columns a to e (here 0 to 4).
fn a1() -> Vec<Vec<f32>> {
    vec![
        vec![0.20, 0.60, 0.75, 0.90, 0.70],
        vec![0.55, 0.30, 0.32, 0.80, 0.85],
        vec![0.70, 0.65, 0.28, 0.90, 0.95],
        vec![0.21, 0.70, 0.90, 0.85, 0.60],
    ]
}

#[test]
fn c1_distance_table() {
    let d1 = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
    let d2 = vec![vec![3.0, 4.0], vec![1.0, 1.0], vec![0.0, 1.0]];
    let t = distance_table(&d1, &d2);
    assert_eq!((t.len(), t[0].len()), (2, 3), "2 descriptors in image 1, 3 in image 2: the table is 2 rows × 3 columns");
    assert!(close(t[0][0], 5.0), "row 0, column 0: got {}, expected 5", t[0][0]);
    assert!(close(t[1][1], 0.0) && close(t[1][2], 1.0), "row 1: got {:?}, expected [.., 0, 1]", t[1]);
}

#[test]
fn c2_nearest_two_a1() {
    let expected = [(0, 0.20, 0.60), (1, 0.30, 0.32), (2, 0.28, 0.65), (0, 0.21, 0.60)];
    for (i, row) in a1().iter().enumerate() {
        let (j, d1, d2) = nearest_two(row);
        let (ej, e1, e2) = expected[i];
        assert!(j == ej && close(d1, e1) && close(d2, e2), "A1(a) row {}: got ({j}, {d1}, {d2}), expected ({ej}, {e1}, {e2})", i + 1);
    }
    // The second nearest may come before the nearest in the row.
    let (j, d1, d2) = nearest_two(&[0.5, 0.9, 0.1]);
    assert!(j == 2 && close(d1, 0.1) && close(d2, 0.5), "row [0.5, 0.9, 0.1]: got ({j}, {d1}, {d2}), expected (2, 0.1, 0.5)");
}

#[test]
fn c3_ratio_test_a1_a2() {
    let got = ratio_test(&a1(), 0.8);
    assert_eq!(got, vec![(0, 0), (2, 2), (3, 0)], "A1(b): rows 1, 3, 4 pass (1–a, 3–c, 4–a)");
    assert_eq!(ratio_test(&a1(), f32::INFINITY).len(), 4, "with no threshold every row has its nearest neighbour");
    // A2: only (0.20, 0.60) passes, at 0.8 and at 0.6.
    let table = vec![vec![0.20, 0.60], vec![0.30, 0.32], vec![0.45, 0.50], vec![0.10, 0.11]];
    for t in [0.8, 0.6] {
        assert_eq!(ratio_test(&table, t), vec![(0, 0)], "A2 at t = {t}: only the first pair passes");
    }
}

#[test]
fn c4_mutual_check_a1_b6() {
    let table = a1();
    let got = mutual_check(&table, &ratio_test(&table, 0.8));
    assert_eq!(got, vec![(0, 0), (2, 2)], "A1(c): 4–a fails (a's nearest is 1), kept 1–a and 3–c");
    // B6: C = [[1, 2], [1.5, 10]]: both rows pick column 0; only (0, 0) is mutual.
    let c = vec![vec![1.0, 2.0], vec![1.5, 10.0]];
    let got = mutual_check(&c, &ratio_test(&c, f32::INFINITY));
    assert_eq!(got, vec![(0, 0)], "B6: only (1, 1) is mutual");
}

#[test]
fn c5_precision_recall_a3() {
    // A3, setting 1: 400 possible, 250 accepted, 210 correct.
    let mut truth: Vec<Option<usize>> = (0..400).map(Some).collect();
    truth.extend(std::iter::repeat_n(None, 100));
    let mut matches: Vec<(usize, usize)> = (0..210).map(|i| (i, i)).collect();
    matches.extend((210..250).map(|i| (i, i + 1)));
    let (correct, p, r) = precision_recall(&matches, &truth);
    assert!(correct == 210 && close(p, 0.84) && close(r, 0.525), "A3 setting 1: got {correct} correct, precision {p}, recall {r}; expected 210, 0.84, 0.525");
    // Setting 2: 150 accepted, 145 correct (the 5 wrong ones start at points without a partner).
    let mut matches: Vec<(usize, usize)> = (0..145).map(|i| (i, i)).collect();
    matches.extend((450..455).map(|i| (i, 0)));
    let (correct, p, r) = precision_recall(&matches, &truth);
    assert!(correct == 145 && (p - 0.967).abs() < 1e-3 && close(r, 0.3625), "A3 setting 2: got {correct}, {p}, {r}; expected 145, 0.97, 0.36");
    let (_, p, _) = precision_recall(&[], &truth);
    assert_eq!(p, 0.0, "nothing accepted: precision 0");
}
