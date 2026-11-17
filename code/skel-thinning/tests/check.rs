// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p skel-thinning
//
// The exercise writes pixels as (row, column); the code indexes img[(x, y)] = img[(column, row)].

use imgcourse::Image;
use skel_thinning::*;

/// A binary image from rows of '0' and '1'.
fn grid(rows: &[&str]) -> Image<bool> {
    let w = rows[0].len();
    Image::from_fn(w, rows.len(), |x, y| rows[y].as_bytes()[x] == b'1')
}

/// The object pixels as (row, column), row by row.
fn pixels(img: &Image<bool>) -> Vec<(usize, usize)> {
    img.coords().filter(|&p| img[p]).map(|(x, y)| (y, x)).collect()
}

fn b1() -> Image<bool> {
    grid(&["0000000", "0000000", "0111100", "0000010", "0000000", "0100000", "0000000"])
}

#[test]
fn c1_end_point_template_b1() {
    let found = hit_or_miss(&b1(), &END_POINT);
    assert!(pixels(&found).is_empty(), "B1(a) the end-point template with the neighbour above finds nothing, got {:?}", pixels(&found));
    let found = hit_or_miss(&b1(), &ISOLATED);
    assert_eq!(pixels(&found), [(5, 1)], "B1(c) the isolated-pixel template finds (row, column) = (5, 1) only");
    let corner = grid(&["000", "011", "011"]);
    let t = [[0, 0, 0], [0, 1, 1], [0, 1, 1]];
    assert!(hit_or_miss(&corner, &t)[(1, 1)], "a template with only 0s and 1s fits its own picture");
    let dont_care = [[DONT_CARE; 3], [DONT_CARE, 1, DONT_CARE], [DONT_CARE; 3]];
    assert_eq!(hit_or_miss(&corner, &dont_care), corner, "a template with only the centre 1 finds every object pixel");
}

#[test]
fn c2_rotate45() {
    let r = rotate45(&END_POINT);
    assert_eq!(r, [[0, 0, 1], [0, 1, 0], [0, 0, 0]], "the end-point template turned 45° clockwise: neighbour up-right");
    assert_eq!(turn(&L2, 2), [[DONT_CARE, 1, DONT_CARE], [1, 1, 0], [DONT_CARE, 0, 0]], "L2 turned 90°: rows ·1· / 110 / ·00");
    assert_eq!(turn(&L1, 8), L1, "8 turns of 45° give the template back");
}

#[test]
fn c2_all_end_points_b1() {
    let found = hit_or_miss_all_turns(&b1(), &END_POINT);
    assert_eq!(pixels(&found), [(2, 1), (3, 5)], "B1(b) all 8 turns find the two ends of the line");
    assert!(hit_or_miss(&b1(), &turn(&END_POINT, 2))[(1, 2)], "B1(b) (2, 1) is found by turn 3, neighbour to the right");
    assert!(hit_or_miss(&b1(), &turn(&END_POINT, 7))[(5, 3)], "B1(b) (3, 5) is found by turn 8, neighbour up-left");
}

#[test]
fn c3_simple_b2() {
    let cases = [
        (["010", "010", "010"], false, "(a) the middle of a line"),
        (["110", "110", "000"], true, "(b) the corner of a block"),
        (["111", "111", "111"], false, "(c) an interior pixel"),
        (["010", "110", "000"], true, "(d) neighbours left and above"),
        (["101", "010", "000"], false, "(e) two diagonal neighbours"),
    ];
    for (rows, simple, what) in cases {
        assert_eq!(is_simple(&grid(&rows), 1, 1), simple, "B2{what}: simple should be {simple}");
    }
    assert!(!is_simple(&grid(&["000", "010", "000"]), 1, 1), "an isolated pixel is not simple");
    assert!(is_simple(&grid(&["000", "010", "010"]), 1, 1), "an end point is simple");
}

/// The 3 × 7 rectangle of B3 in a 5 × 9 image.
fn rectangle() -> Image<bool> {
    Image::from_fn(9, 5, |x, y| (1..=7).contains(&x) && (1..=3).contains(&y))
}

#[test]
fn c4_thin_rectangle_b3() {
    let (skel, passes) = thin(&rectangle(), false);
    let mut expected: Vec<(usize, usize)> = (1..=6).map(|c| (2, c)).collect();
    expected.extend([(1, 1), (3, 1), (1, 7)]);
    expected.sort();
    let mut got = pixels(&skel);
    got.sort();
    assert_eq!(got, expected, "B3(c) the skeleton of the rectangle, as (row, column)");
    assert_eq!(passes, 2, "B3(c) two passes remove pixels, the third removes nothing");
}

#[test]
fn c4_parallel_cuts_the_line_b4() {
    let line = Image::from_fn(12, 4, |x, y| (1..=10).contains(&x) && (1..=2).contains(&y));
    let (par, _) = thin(&line, true);
    assert!(pixels(&par).is_empty(), "B4(b) in parallel both rows go: got {:?}", pixels(&par));
    let (seq, _) = thin(&line, false);
    let mut expected: Vec<(usize, usize)> = (2..=9).map(|c| (2, c)).collect();
    expected.extend([(1, 1), (1, 10)]);
    expected.sort();
    let mut got = pixels(&seq);
    got.sort();
    assert_eq!(got, expected, "B4(b) one template at a time keeps a one-pixel line");
}

#[test]
fn c5_simple_points_b5() {
    // A 7 × 7 square with a 3 × 3 hole, in a 9 × 9 image.
    let ring = Image::from_fn(9, 9, |x, y| (1..=7).contains(&x) && (1..=7).contains(&y) && !((3..=5).contains(&x) && (3..=5).contains(&y)));
    let skel = thin_simple(&ring, false);
    let n = pixels(&skel).len();
    assert!(n >= 8 && !skel[(4, 4)], "B5(b) the ring keeps its hole: got {n} pixels");
    assert!(skel.coords().filter(|&p| skel[p]).all(|(x, y)| !is_simple(&skel, x, y)), "B5(c) no pixel of the loop is simple");
    let point = thin_simple(&rectangle(), false);
    assert_eq!(pixels(&point).len(), 1, "B5(c) without keeping end points the rectangle shrinks to one pixel");
    let line = thin_simple(&rectangle(), true);
    let rows: Vec<usize> = pixels(&line).iter().map(|p| p.0).collect();
    assert!(pixels(&line).len() >= 5 && rows.iter().all(|&r| r == 2), "keeping end points leaves a line along row 2: got {:?}", pixels(&line));
}
