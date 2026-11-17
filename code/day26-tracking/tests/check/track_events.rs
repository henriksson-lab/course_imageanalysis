// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day26 track_events::

use day26::track_events::*;

fn det(t: usize, x: f32, y: f32, area: f32) -> Detection {
    Detection { t, x, y, area }
}

#[test]
fn c1_gap_b2() {
    // B2: object a at (11, 11) in frame 2 and (13, 12) in frame 4: d² = 5 over g = 2 frames
    let (end, start) = (det(2, 11.0, 11.0, 102.0), det(4, 13.0, 12.0, 104.0));
    let c = gap_cost(&end, &start, 2, 3.0);
    assert!(c.is_some_and(|c| (c - 2.5).abs() < 1e-5), "B2(c): gap cost {c:?}, expected 2.5");
    assert!(gap_cost(&end, &start, 0, 3.0).is_none(), "max_gap 0 allows no gap at all");
    assert!(gap_cost(&end, &det(3, 12.0, 11.0, 100.0), 2, 3.0).is_none(), "g = 1 is a normal link, not a gap");
    assert!(gap_cost(&start, &end, 2, 3.0).is_none(), "a start before the end is not a gap");
}

#[test]
fn c1_gate_grows_with_the_gap() {
    // gate 1: radius 1 for one frame, √2 for g = 2, √3 for g = 3 (the lecture's figure)
    let end = det(0, 0.0, 0.0, 1.0);
    assert!(gap_cost(&end, &det(2, 1.4, 0.0, 1.0), 5, 1.0).is_some(), "1.4 away after g = 2 is inside √2 = 1.41");
    assert!(gap_cost(&end, &det(2, 1.5, 0.0, 1.0), 5, 1.0).is_none(), "1.5 away after g = 2 is outside √2 = 1.41");
    assert!(gap_cost(&end, &det(3, 1.5, 0.0, 1.0), 5, 1.0).is_some(), "1.5 away after g = 3 is inside √3 = 1.73");
}

#[test]
fn c2_split_b2() {
    // B2: the mother (42, 13), area 222, in frame 3; daughters (48, 14, 110) and (36, 13, 112)
    let mother = det(3, 42.0, 13.0, 222.0);
    let c = split_cost(&mother, &det(4, 48.0, 14.0, 110.0), &det(4, 36.0, 13.0, 112.0));
    assert!((c - 37.0).abs() < 1e-3, "B2: split cost {c}, expected 37 (d² = 37, area factor 1)");
    // daughters that are together only half the mother's area: ρ = 2, factor 4
    let c = split_cost(&mother, &det(4, 48.0, 14.0, 60.0), &det(4, 36.0, 13.0, 51.0));
    assert!((c - 148.0).abs() < 1e-2, "areas 60 + 51 for a mother of 222: cost {c}, expected 37 · 4 = 148");
}

#[test]
fn c3_link_scores_b4() {
    // B4: 12 reference links; 11 found, 10 of them right
    let reference: Vec<(usize, usize)> = (0..12).map(|i| (i, i + 100)).collect();
    let mut result: Vec<(usize, usize)> = (0..10).map(|i| (i, i + 100)).collect();
    result.push((10, 999));
    let (p, r, f1) = link_scores(&reference, &result);
    assert!((p - 10.0 / 11.0).abs() < 1e-4, "B4(a): precision {p}, expected 10/11 = 0.909");
    assert!((r - 10.0 / 12.0).abs() < 1e-4, "B4(a): recall {r}, expected 10/12 = 0.833");
    assert!((f1 - 20.0 / 23.0).abs() < 1e-4, "B4(a): F1 {f1}, expected 20/23 = 0.870");
}

#[test]
fn c4_division_scores_b4() {
    // B4(b): right mother, one wrong daughter: TP 0, FP 1, FN 1
    let reference = [[5, 6, 7]];
    assert_eq!(division_scores(&reference, &[[5, 6, 9]]), (0, 1, 1), "B4(b): one wrong daughter");
    assert_eq!(division_scores(&reference, &[[5, 7, 6]]), (1, 0, 0), "the daughters may come in either order");
    assert_eq!(division_scores(&reference, &[]), (0, 0, 1), "nothing found: one false negative");
}

#[test]
fn c5_cycle_times_b3() {
    // B3: the lineage tree of the 48 h movie
    let cell = |born: Option<f32>, divided: Option<f32>| Cell { born, divided };
    let cells = [
        cell(None, Some(10.0)),       // F, present at the start
        cell(Some(10.0), Some(28.0)), // A
        cell(Some(10.0), Some(34.0)), // B
        cell(Some(28.0), Some(46.0)), // C
        cell(Some(28.0), None),       // D
        cell(Some(34.0), None),       // E
        cell(Some(34.0), None),       // G
        cell(Some(46.0), None),       // H
        cell(Some(46.0), None),       // I
    ];
    let (complete, censored) = cycle_times(&cells, 48.0);
    let mut c = complete.clone();
    c.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(c, vec![18.0, 18.0, 24.0], "B3(a): complete cycles of A, B and C");
    let mut s = censored.clone();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(s, vec![2.0, 2.0, 14.0, 14.0, 20.0], "B3(a): observed times of the censored cells D, E, G, H, I");
    let mean_all = (complete.iter().sum::<f32>() + censored.iter().sum::<f32>()) / 8.0;
    assert!((mean_all - 14.0).abs() < 1e-4, "B3(c): mean of all observed times {mean_all}, expected 14");
}
