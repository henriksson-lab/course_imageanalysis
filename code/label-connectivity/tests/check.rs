// Checks for the coding exercises, using the masks of the pen-and-paper exercises.
// Run: cargo test -p label-connectivity
// The exercise writes pixels as (row, column); the code uses (x, y) = (column, row).

use imgcourse::Image;
use label_connectivity::*;

/// A mask written row by row, "1" = object, anything else = background.
fn mask(rows: &[&str]) -> Image<bool> {
    let cells: Vec<Vec<bool>> = rows.iter().map(|r| r.split_whitespace().map(|c| c == "1").collect()).collect();
    Image::from_fn(cells[0].len(), cells.len(), |x, y| cells[y][x])
}

/// A label image written row by row ("." = 0).
fn labels(rows: &[&str]) -> Image<u32> {
    let cells: Vec<Vec<u32>> = rows.iter().map(|r| r.split_whitespace().map(|c| c.parse().unwrap_or(0)).collect()).collect();
    Image::from_fn(cells[0].len(), cells.len(), |x, y| cells[y][x])
}

fn b1() -> Image<bool> {
    mask(&[
        "1 1 0 0 0 0 1 0",
        "1 1 0 0 0 1 0 0",
        "0 0 1 0 1 0 1 0",
        "0 0 0 1 0 0 0 1",
        "0 1 1 1 0 0 0 0",
        "0 1 0 1 0 0 1 1",
        "0 1 1 1 0 0 1 1",
        "0 0 0 0 0 0 0 0",
    ])
}

fn b2() -> Image<bool> {
    mask(&["1 0 0 0 1 0", "1 0 0 1 0 0", "1 0 0 1 0 1", "1 1 1 1 0 1", "0 0 0 0 0 1", "1 1 0 1 1 1"])
}

fn same(got: &Image<u32>, expected: &Image<u32>, what: &str) {
    for (x, y) in got.coords() {
        assert_eq!(got[(x, y)], expected[(x, y)], "{what}, row {y}, column {x}: got {}, expected {}\ngot:\n{got}expected:\n{expected}", got[(x, y)], expected[(x, y)]);
    }
}

#[test]
fn c1_queue_fills_ring_by_ring() {
    // A 7 × 7 block, 4-connectivity, seed at the top left (the lecture's figure): with a queue,
    // pixels come out in order of their distance x + y from the seed (exercise B8).
    let block = Image::new(7, 7, true);
    let mut lab = Image::new(7, 7, 0u32);
    let order = flood_fill(&block, &mut lab, (0, 0), 1, false);
    assert_eq!(order.len(), 49, "every pixel once: got {} pixels", order.len());
    assert_eq!(order[0], (0, 0), "the seed comes out first");
    for w in order.windows(2) {
        let (a, b) = (w[0].0 + w[0].1, w[1].0 + w[1].1);
        assert!(a <= b, "{:?} (distance {a}) came out before {:?} (distance {b})", w[0], w[1]);
    }
    assert!(lab.pixels().iter().all(|&l| l == 1), "every pixel must get label 1");
}

#[test]
fn c1_stops_at_the_background() {
    // Two blocks that touch only at a corner: one component with 8, two with 4.
    let m = mask(&["1 1 0 0", "1 1 0 0", "0 0 1 1", "0 0 1 1"]);
    let mut lab = Image::new(4, 4, 0u32);
    let n4 = flood_fill(&m, &mut lab, (0, 0), 1, false).len();
    assert_eq!(n4, 4, "4-connectivity from (0, 0): got {n4} pixels, expected 4");
    let mut lab = Image::new(4, 4, 0u32);
    let n8 = flood_fill(&m, &mut lab, (0, 0), 1, true).len();
    assert_eq!(n8, 8, "8-connectivity from (0, 0): got {n8} pixels, expected 8");
}

#[test]
fn c2_count_b1() {
    let (_, n8) = label_flood_fill(&b1(), true);
    assert_eq!(n8, 2, "B1 (a), 8-connectivity: got {n8} objects, expected 2");
    let (_, n4) = label_flood_fill(&b1(), false);
    assert_eq!(n4, 9, "B1 (b), 4-connectivity: got {n4} objects, expected 9");
}

#[test]
fn c2_holes_b1() {
    // B1 (c): the background with 4-connectivity has 4 components.
    let background = b1().map(|v| !v);
    let (_, n) = label_flood_fill(&background, false);
    assert_eq!(n, 4, "B1 (c), background components with 4-connectivity: got {n}, expected 4");
    let (_, n) = label_flood_fill(&background, true);
    assert_eq!(n, 2, "B1 (c), background components with 8-connectivity: got {n}, expected 2");
}

#[test]
fn c2_final_labels_b2() {
    let (lab, n) = label_flood_fill(&b2(), true);
    assert_eq!(n, 3, "B2: got {n} objects, expected 3");
    let expected = labels(&["1 . . . 1 .", "1 . . 1 . .", "1 . . 1 . 2", "1 1 1 1 . 2", ". . . . . 2", "3 3 . 2 2 2"]);
    same(&lab, &expected, "B2 (b)");
    let (_, n4) = label_flood_fill(&b2(), false);
    assert_eq!(n4, 4, "B2 (c), 4-connectivity: got {n4} objects, expected 4");
}

#[test]
fn c3_naive_b3() {
    let mut uf = UnionFind::new(8); // elements 0..=7; 0 is not used
    for (a, b) in [(1, 2), (3, 4), (2, 4), (5, 6), (6, 7), (4, 7)] {
        uf.union(a, b);
    }
    assert_eq!(uf.parent[1..], [2, 4, 4, 7, 6, 7, 7], "B3 (a): parents of 1..7");
    assert_eq!(uf.find(1), 7, "B3 (a): find(1)");
}

#[test]
fn c3_by_rank_b3() {
    let mut uf = UnionFind::new(8);
    for (a, b) in [(1, 2), (3, 4), (2, 4), (5, 6), (6, 7), (4, 7)] {
        uf.union_by_rank(a, b);
    }
    assert_eq!(uf.parent[1..], [2, 4, 4, 4, 6, 4, 6], "B3 (c): parents of 1..7");
    assert_eq!((uf.rank[2], uf.rank[4], uf.rank[6]), (1, 2, 1), "B3 (c): ranks of 2, 4, 6");
    assert_eq!(uf.find(1), 4, "B3 (c): find(1)");
}

#[test]
fn c4_provisional_labels_b2() {
    let (prov, _, _) = label_two_pass(&b2(), true);
    let expected = labels(&["1 . . . 2 .", "1 . . 2 . .", "1 . . 2 . 3", "1 1 1 1 . 3", ". . . . . 3", "4 4 . 5 3 3"]);
    same(&prov, &expected, "B2 (a), first pass");
}

#[test]
fn c4_final_labels_b2() {
    let (_, lab, n) = label_two_pass(&b2(), true);
    assert_eq!(n, 3, "B2 (b): got {n} objects, expected 3");
    let expected = labels(&["1 . . . 1 .", "1 . . 1 . .", "1 . . 1 . 2", "1 1 1 1 . 2", ". . . . . 2", "3 3 . 2 2 2"]);
    same(&lab, &expected, "B2 (b)");
}

#[test]
fn c4_same_as_flood_fill() {
    // The micrograph's dark regions: both algorithms must give the same label image.
    let m = imgcourse::samples::cell().crop(60, 60, 100, 100).map(|v| v < 0.37);
    for eight in [false, true] {
        let (a, na) = label_flood_fill(&m, eight);
        let (_, b, nb) = label_two_pass(&m, eight);
        assert_eq!(na, nb, "eight = {eight}: flood fill counts {na}, two passes {nb}");
        same(&b, &a, "two passes against flood fill");
    }
}
