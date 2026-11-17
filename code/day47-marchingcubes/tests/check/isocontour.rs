// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day47 isocontour::

use imgcourse::Image;
use day47::isocontour::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// The 4 × 4 grid of B1 (c = 5).
fn b1() -> Image<f32> {
    Image::from_rows(&[[4.0, 9.0, 3.0, 7.0], [3.0, 1.0, 9.0, 6.0], [2.0, 4.0, 8.0, 3.0], [2.0, 1.0, 3.0, 4.0]])
}

/// B5: the grid of B1 with a frame of zeros.
fn b5() -> Image<f32> {
    let g = b1();
    Image::from_fn(6, 6, |x, y| if (1..5).contains(&x) && (1..5).contains(&y) { g[(x - 1, y - 1)] } else { 0.0 })
}

#[test]
fn c1_cases_b1() {
    let g = b1();
    let expected = [[2, 5, 14], [0, 6, 11], [0, 2, 1]]; // [row][column] of cells
    for y in 0..3 {
        for x in 0..3 {
            let corners = [g[(x, y)], g[(x + 1, y)], g[(x + 1, y + 1)], g[(x, y + 1)]];
            let case = case_index(corners, 5.0);
            assert_eq!(case, expected[y][x], "B1(a): cell ({x}, {y}) with corners {corners:?}: got case {case}, expected {}", expected[y][x]);
        }
    }
}

#[test]
fn c2_crossing_b2() {
    for (c, t) in [(120.0, 0.5), (60.0, 0.125), (180.0, 0.875)] {
        let got = crossing(40.0, 200.0, c);
        assert!(close(got, t, 1e-6), "B2(a): from 40 to 200 at c = {c}: got t = {got}, expected {t}");
    }
    assert!(close(crossing(9.0, 3.0, 5.0), 2.0 / 3.0, 1e-6), "B1(c): from 9 to 3 at c = 5: expected t = 0.667");
}

#[test]
fn c3_decider_b3_b1() {
    let f = saddle_value([8.0, 4.0, 7.0, 3.0]);
    assert!(close(f, 5.5, 1e-5), "B3(b): corners 8, 4, 7, 3: got f* = {f}, expected 5.5");
    let f = saddle_value([6.0, 4.0, 6.5, 3.0]);
    assert!(close(f, 4.909, 1e-3), "B3(c): corners 6, 4, 6.5, 3: got f* = {f}, expected 4.91");
    let f = saddle_value([9.0, 3.0, 9.0, 1.0]);
    assert!(close(f, 5.571, 1e-3), "B1(b): the saddle cell (1, 0): got f* = {f}, expected 78/14 = 5.57");
}

#[test]
fn c4_crossing_points_b1() {
    let segs = marching_squares(&b1(), 5.0, true);
    assert_eq!(segs.len(), 8, "B1: 7 cells with a contour, the saddle with 2 segments: expected 8 segments, got {}", segs.len());
    // B1(c): every crossing point of the table must be an end of some segment.
    let table = [(0.2, 0.0), (1.667, 0.0), (2.5, 0.0), (1.0, 0.5), (2.0, 0.333), (1.5, 1.0), (3.0, 1.333), (1.25, 2.0), (2.6, 2.0), (2.0, 2.6)];
    for (x, y) in table {
        let found = segs.iter().any(|&(p, q)| (close(p.0, x, 2e-3) && close(p.1, y, 2e-3)) || (close(q.0, x, 2e-3) && close(q.1, y, 2e-3)));
        assert!(found, "B1(c): the crossing point ({x}, {y}) is not an end of any segment");
    }
    // B1(b): with the decider the saddle cell joins the two 9s: its segments cut off the dark
    // corners, (1.667, 0)–(2, 0.333) around the 3 and (1, 0.5)–(1.5, 1) around the 1.
    let joined = |a: (f32, f32), b: (f32, f32)| segs.iter().any(|&(p, q)| {
        let is = |u: (f32, f32), v: (f32, f32)| close(u.0, v.0, 2e-3) && close(u.1, v.1, 2e-3);
        (is(p, a) && is(q, b)) || (is(p, b) && is(q, a))
    });
    assert!(joined((1.667, 0.0), (2.0, 0.333)) && joined((1.0, 0.5), (1.5, 1.0)), "B1(b): the decider joins: in the saddle cell (1.667, 0) must be connected to (2, 0.333), and (1, 0.5) to (1.5, 1)");
    let separate = marching_squares(&b1(), 5.0, false);
    let sep_joined = separate.iter().any(|&(p, q)| (close(p.0, 1.667, 2e-3) && close(q.0, 1.0, 2e-3)) || (close(q.0, 1.667, 2e-3) && close(p.0, 1.0, 2e-3)));
    assert!(sep_joined, "B1(e): separate connects (1.667, 0) to (1, 0.5)");
}

#[test]
fn c4_orientation_one_bright_pixel() {
    // One bright pixel: a diamond around it, walked with the bright pixel on the right.
    let mut img = Image::new(3, 3, 0.0f32);
    img[(1, 1)] = 1.0;
    let segs = marching_squares(&img, 0.5, true);
    assert_eq!(segs.len(), 4, "one bright pixel: expected 4 segments, got {}", segs.len());
    for &(p, q) in &segs {
        let cross = (q.0 - p.0) * (1.0 - p.1) - (q.1 - p.1) * (1.0 - p.0);
        assert!(cross > 0.0, "segment {p:?} → {q:?} must have the bright pixel (1, 1) on its right");
    }
    // Each end must be the start of the next segment: a closed, consistently oriented contour.
    for &(_, q) in &segs {
        assert!(segs.iter().any(|&(p, _)| close(p.0, q.0, 1e-5) && close(p.1, q.1, 1e-5)), "the end {q:?} is not the start of another segment");
    }
}

#[test]
fn c5_shoelace_b4() {
    // The diamond of B4 as four segments, and reversed.
    let pts = [(1.0, 0.0), (2.0, 1.0), (1.0, 2.0), (0.0, 1.0)];
    let segs: Vec<Segment> = (0..4).map(|i| (pts[i], pts[(i + 1) % 4])).collect();
    let rev: Vec<Segment> = segs.iter().map(|&(p, q)| (q, p)).collect();
    assert!(close(enclosed_area(&segs), 2.0, 1e-5), "B4(a): area {}, expected 2", enclosed_area(&segs));
    assert!(close(enclosed_area(&rev), -2.0, 1e-5), "B4(b): reversed: area {}, expected −2", enclosed_area(&rev));
    assert!(close(contour_length(&segs), 4.0 * 2f32.sqrt(), 1e-5), "B4(a): perimeter {}, expected 4√2", contour_length(&segs));
}

#[test]
fn c5_decider_against_separate_b5() {
    let dec = marching_squares(&b5(), 5.0, true);
    let sep = marching_squares(&b5(), 5.0, false);
    let (ld, ls) = (contour_length(&dec), contour_length(&sep));
    let (ad, as_) = (enclosed_area(&dec), enclosed_area(&sep));
    assert!(close(ld, 10.42, 0.01) && close(ls, 10.90, 0.01), "B5(b): lengths: decider {ld:.2}, separate {ls:.2}; expected 10.42 and 10.90");
    assert!(close(ad, 4.53, 0.01) && close(as_, 0.69 + 3.35, 0.01), "B5(a): areas: decider {ad:.2}, separate {as_:.2}; expected 4.53 and 0.69 + 3.35 = 4.04");
}
