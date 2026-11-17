// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p dynprog dp_paths::
//
// The exercise writes pixels as (row, column); the code as (x, y) = (column, row).

use dynprog::dp_paths::*;
use imgcourse::Image;

/// The cost image of B1 (the lecture's example).
fn b1() -> Image<f32> {
    Image::from_rows(&[
        [7., 4., 8., 8., 3.],
        [2., 9., 3., 8., 2.],
        [3., 1., 3., 6., 8.],
        [5., 1., 7., 3., 2.],
        [9., 2., 3., 9., 6.],
    ])
}

#[test]
fn c1_table_b1() {
    let (m, pointer) = dp_table(&b1());
    let expected = [
        [7., 4., 8., 8., 3.],
        [6., 13., 7., 11., 5.],
        [9., 7., 10., 11., 13.],
        [12., 8., 14., 13., 13.],
        [17., 10., 11., 22., 19.],
    ];
    for (y, row) in expected.iter().enumerate() {
        for (x, &e) in row.iter().enumerate() {
            assert_eq!(m[(x, y)], e, "M at row {y}, column {x}: got {}, expected {e}", m[(x, y)]);
        }
    }
    // M(1, 0) = 2 + min(7, 4): the pointer goes to (row 0, column 1), one column right.
    assert_eq!(pointer[(0, 1)], 1, "pointer at row 1, column 0: got {}, expected +1", pointer[(0, 1)]);
    // M(2, 1) = 1 + min(6, 13, 7): from (row 1, column 0), one column left.
    assert_eq!(pointer[(1, 2)], -1, "pointer at row 2, column 1: got {}, expected -1", pointer[(1, 2)]);
}

#[test]
fn c2_backtrack_b1() {
    let cost = b1();
    let (m, pointer) = dp_table(&cost);
    let path = backtrack(&m, &pointer);
    assert_eq!(path, vec![1, 0, 1, 1, 1], "the cheapest path (columns, top to bottom): got {path:?}");
    assert_eq!(path_cost(&cost, &path), 10.0, "its cost: got {}, expected 10", path_cost(&cost, &path));
}

#[test]
fn c2_change_e6() {
    // E6: raise C(0, 1) from 4 to 9: the best path now starts at the 7 and costs 13.
    let mut cost = b1();
    cost[(1, 0)] = 9.0;
    let (m, pointer) = dp_table(&cost);
    let path = backtrack(&m, &pointer);
    assert_eq!(path, vec![0, 0, 1, 1, 1], "E6 path: got {path:?}");
    assert_eq!(path_cost(&cost, &path), 13.0, "E6 cost: got {}, expected 13", path_cost(&cost, &path));
}

#[test]
fn c3_greedy_b1() {
    let cost = b1();
    let path = greedy_path(&cost);
    assert_eq!(path, vec![4, 4, 3, 4, 4], "greedy path (columns): got {path:?}");
    assert_eq!(path_cost(&cost, &path), 19.0, "greedy cost: got {}, expected 19", path_cost(&cost, &path));
}

#[test]
fn c3_predict_b5() {
    // B5: C(0, 4) = 1. The best path does not change; greedy now costs 17.
    let mut cost = b1();
    cost[(4, 0)] = 1.0;
    let (m, pointer) = dp_table(&cost);
    assert_eq!(backtrack(&m, &pointer), vec![1, 0, 1, 1, 1], "B5: the best path should not change");
    let g = path_cost(&cost, &greedy_path(&cost));
    assert_eq!(g, 17.0, "B5 greedy cost: got {g}, expected 17");
}

#[test]
fn c4_count_b2() {
    let n = count_paths(5, 5);
    let expected = [[1., 1., 1., 1., 1.], [2., 3., 3., 3., 2.], [5., 8., 9., 8., 5.], [13., 22., 25., 22., 13.], [35., 60., 69., 60., 35.]];
    for (y, row) in expected.iter().enumerate() {
        for (x, &e) in row.iter().enumerate() {
            assert_eq!(n[(x, y)], e, "paths to row {y}, column {x}: got {}, expected {e}", n[(x, y)]);
        }
    }
    let total: f64 = (0..5).map(|x| n[(x, 4)]).sum();
    assert_eq!(total, 259.0, "all paths of a 5 × 5 image: got {total}, expected 259");
    // B2 (c): about 1.5 · 10^49 paths through 100 × 100.
    let n = count_paths(100, 100);
    let total: f64 = (0..100).map(|x| n[(x, 99)]).sum();
    assert!(total > 1e49 && total < 2e49, "100 × 100: got {total:e}, expected about 1.5e49");
}

#[test]
fn c5_distance_b4() {
    // A 3 × 4 image (3 rows, 4 columns) whose only background pixel is the top right.
    let bg = Image::from_fn(4, 3, |x, y| (x, y) == (3, 0));
    let d = distance_forward(&bg);
    for y in 0..3 {
        assert_eq!(d[(3, y)], y as f32, "forward pass, row {y}, column 3: got {}, expected {y}", d[(3, y)]);
        for x in 0..3 {
            assert!(d[(x, y)].is_infinite(), "forward pass, row {y}, column {x}: got {}, expected ∞ (B4 a)", d[(x, y)]);
        }
    }
    let d = distance_two_passes(&bg);
    let expected = [[3., 2., 1., 0.], [4., 3., 2., 1.], [5., 4., 3., 2.]];
    for (y, row) in expected.iter().enumerate() {
        for (x, &e) in row.iter().enumerate() {
            assert_eq!(d[(x, y)], e, "two passes, row {y}, column {x}: got {}, expected {e}", d[(x, y)]);
        }
    }
}
