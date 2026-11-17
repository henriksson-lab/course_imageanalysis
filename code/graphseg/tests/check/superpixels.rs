// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p graphseg superpixels::

use imgcourse::Image;
use graphseg::superpixels::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn c1_distance_b1() {
    // pixel I = 50 at (4, 4); A: I = 52 at (0, 0); B: I = 70 at (6, 4); S = 8
    let got = [
        slic_distance2(50.0 - 52.0, 4.0, 4.0, 10.0, 8.0),
        slic_distance2(50.0 - 70.0, -2.0, 0.0, 10.0, 8.0),
        slic_distance2(50.0 - 52.0, 4.0, 4.0, 40.0, 8.0),
        slic_distance2(50.0 - 70.0, -2.0, 0.0, 40.0, 8.0),
    ];
    let expected = [54.0, 406.25, 804.0, 500.0];
    for k in 0..4 {
        assert!(close(got[k], expected[k], 1e-3), "B1: D² to A and B for m = 10, then m = 40: got {got:?}, expected {expected:?}");
    }
}

#[test]
fn c2_init_grid() {
    let flat = Image::new(8, 8, 50.0f32);
    let (c, s) = slic_init(&flat, 4);
    assert!(close(s, 4.0, 1e-6), "S = sqrt(64/4) = 4, got {s}");
    let pos: Vec<(f32, f32)> = c.iter().map(|c| (c.x, c.y)).collect();
    assert_eq!(pos, [(2.0, 2.0), (6.0, 2.0), (2.0, 6.0), (6.0, 6.0)], "grid points at (i + ½)S on a flat image");
    // B2(a): N = 65 536, K = 256 gives S = 16 and 256 centres
    let (c, s) = slic_init(&Image::new(256, 256, 0.0f32), 256);
    assert!(close(s, 16.0, 1e-5) && c.len() == 256, "B2(a): got S = {s} and {} centres, expected 16 and 256", c.len());
}

#[test]
fn c2_init_avoids_an_edge() {
    // a vertical edge between x = 1 and x = 2: the grid point (2, 2) has a large gradient
    let img = Image::from_fn(8, 8, |x, _| if x < 2 { 0.0 } else { 100.0 });
    let (c, _) = slic_init(&img, 4);
    assert!(c[0].x != 2.0 && c[0].x != 1.0, "the first centre should move off the edge, got ({}, {})", c[0].x, c[0].y);
}

#[test]
fn c3_step_follows_an_edge_and_counts_b2() {
    let img = Image::from_fn(16, 16, |x, _| if x < 7 { 0.0 } else { 100.0 });
    let (mut c, s) = slic_init(&img, 4);
    let mut labels = Image::new(16, 16, 0u32);
    for _ in 0..5 {
        slic_step(&img, &mut c, &mut labels, s, 10.0);
    }
    for (x, y) in img.coords() {
        let l = labels[(x, y)];
        let other = labels.pixels().iter().zip(img.pixels()).any(|(&m, &v)| m == l && v != img[(x, y)]);
        assert!(!other, "superpixel {l} contains both sides of the edge (pixel ({x}, {y}))");
    }
    // B2(b): about 4N distances per iteration, far fewer than N·K
    let big = Image::from_fn(256, 256, |x, y| ((x / 7 + y / 5) % 3) as f32 * 30.0);
    let (mut c, s) = slic_init(&big, 256);
    let mut l = Image::new(256, 256, 0u32);
    let n = slic_step(&big, &mut c, &mut l, s, 20.0);
    assert!((3 * 65536..=4 * 65536).contains(&n), "B2(b): got {n} distances, expected a little under 4N = 262 144");
    assert!(l.pixels().iter().all(|&v| v > 0), "after the first step every pixel should have a superpixel");
}

#[test]
fn c4_connectivity() {
    // label 1 has a stray pixel inside label 2; label 2 is split in two pieces
    let labels = Image::from_rows(&[[1u32, 1, 2, 2], [1, 1, 1, 2], [2, 1, 2, 2], [1, 1, 2, 2]]);
    let out = enforce_connectivity(&labels);
    let expected = Image::from_rows(&[[1u32, 1, 2, 2], [1, 1, 1, 2], [1, 1, 2, 2], [1, 1, 2, 2]]);
    assert_eq!(out, expected, "the stray 2 at (0, 2) should join its left/upper neighbour's label:\ngot\n{out}");
    let (_, sizes) = pieces(&out);
    assert_eq!(sizes.len(), 2, "after the clean-up every label is one piece");
}

/// B3: an 8 × 8 image with the object in rows 3–6, columns 4–7 (counted from 1).
fn b3_truth() -> Image<u32> {
    Image::from_fn(8, 8, |x, y| ((2..6).contains(&y) && (3..7).contains(&x)) as u32)
}

#[test]
fn c5_accuracy_and_error_b3() {
    let quadrants = Image::from_fn(8, 8, |x, y| (y / 4 * 2 + x / 4) as u32 + 1);
    let a = achievable_accuracy(&quadrants, &b3_truth());
    let e = undersegmentation_error(&quadrants, &b3_truth());
    assert!(close(a, 0.75, 1e-6), "B3(a): got accuracy {a}, expected 0.75");
    assert!(close(e, 0.5, 1e-6), "B3(b): got under-segmentation error {e}, expected 0.5");
    let band = |v: usize, cuts: [usize; 2]| (v >= cuts[0]) as u32 + (v >= cuts[1]) as u32;
    let shifted = Image::from_fn(8, 8, |x, y| band(y, [2, 6]) * 3 + band(x, [3, 7]) + 1);
    assert!(close(achievable_accuracy(&shifted, &b3_truth()), 1.0, 1e-6), "B3(d): accuracy should be 1");
    assert!(close(undersegmentation_error(&shifted, &b3_truth()), 0.0, 1e-6), "B3(d): error should be 0");
}

#[test]
fn c6_recall_b3() {
    let quadrants = Image::from_fn(8, 8, |x, y| (y / 4 * 2 + x / 4) as u32 + 1);
    let r = boundary_recall(&quadrants, &b3_truth(), 0);
    assert!(close(r, 0.0, 1e-6), "B3(c): got recall {r}, expected 0/16");
    let band = |v: usize, cuts: [usize; 2]| (v >= cuts[0]) as u32 + (v >= cuts[1]) as u32;
    let shifted = Image::from_fn(8, 8, |x, y| band(y, [2, 6]) * 3 + band(x, [3, 7]) + 1);
    let r = boundary_recall(&shifted, &b3_truth(), 0);
    assert!(close(r, 1.0, 1e-6), "B3(d): got recall {r}, expected 16/16");
    // with a tolerance of 2 pixels the left, top and bottom sides are found (1 or 2 pixels from
    // a quadrant boundary), the right side is 3 pixels away
    let r = boundary_recall(&quadrants, &b3_truth(), 2);
    assert!(close(r, 0.75, 1e-6), "B3 with a tolerance of 2 pixels: got recall {r}, expected 12/16");
}
