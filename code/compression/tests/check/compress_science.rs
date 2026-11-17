// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p compression compress_science::

use compression::compress_science::*;
use imgcourse::{Image, Rng};

#[test]
fn c1_lift53_by_hand() {
    // even samples 10, 14, 15, 16; odd samples 12, 13, 17, 18
    let (s, d) = lift53(&[10, 12, 14, 13, 15, 17, 16, 18]);
    assert_eq!(d, vec![0, -1, 2, 2], "d[n] = x[2n+1] − floor((x[2n] + x[2n+2]) / 2), mirrored at the end");
    assert_eq!(s, vec![10, 14, 15, 17], "s[n] = x[2n] + floor((d[n−1] + d[n] + 2) / 4), mirrored at the start");
}

#[test]
fn c1_constant_has_no_detail() {
    let (s, d) = lift53(&[7; 10]);
    assert_eq!(d, vec![0; 5], "a constant signal: every detail is 0");
    assert_eq!(s, vec![7; 5], "a constant signal: the smooth half is the same constant");
}

#[test]
fn c2_exactly_reversible_b7() {
    let mut rng = Rng::new(5);
    for len in [2, 4, 6, 16, 50] {
        let x: Vec<i32> = (0..len).map(|_| rng.below(4096) as i32 - 2048).collect();
        let (s, d) = lift53(&x);
        let back = unlift53(&s, &d);
        assert_eq!(back, x, "length {len}: unlift53(lift53(x)) must give x back exactly");
    }
}

#[test]
fn c3_wavelet2d_quarters() {
    let flat = Image::new(8, 6, 100);
    let w = wavelet2d(&flat);
    for (x, y) in w.coords() {
        let e = if x < 4 && y < 3 { 100 } else { 0 };
        assert_eq!(w[(x, y)], e, "a flat image of 100: ({x}, {y}) should be {e} (LL top left, details 0), got {}", w[(x, y)]);
    }
    // A vertical edge: every row is the same, so the columns have no detail (bottom half 0).
    let edge = Image::from_fn(8, 8, |x, _| if x < 3 { 0 } else { 50 });
    let w = wavelet2d(&edge);
    for (x, y) in w.coords().filter(|&(_, y)| y >= 4) {
        assert_eq!(w[(x, y)], 0, "vertical edge: ({x}, {y}) in the bottom half should be 0, got {}", w[(x, y)]);
    }
    assert!((0..4).any(|y| w[(5, y)] != 0), "vertical edge: the top-right (horizontal detail) quarter should see the edge");
}

#[test]
fn c4_bitplanes_b6() {
    for (planes, expected) in [(1, 47.5), (2, 39.5), (3, 43.5), (6, 45.0)] {
        let g = bitplane_guess(45, 6, planes);
        assert_eq!(g, expected, "B6, 45 = 101101 after {planes} planes: got {g}, expected {expected}");
    }
}

#[test]
fn c5_chunks_b4() {
    let region = [(100, 612), (300, 812), (40, 41)];
    let n = chunks_read(&region, &[256, 256, 1]);
    assert_eq!(n, 9, "B4(a), chunks 256 × 256 × 1: got {n}, expected 9");
    let n = chunks_read(&region, &[64, 64, 64]);
    assert_eq!(n, 81, "B4(b), chunks 64³: got {n}, expected 81");
    // the lecture's x–z plane of a 2048 × 2048 × 100 stack
    let plane = [(0, 2048), (700, 701), (0, 100)];
    assert_eq!(chunks_read(&plane, &[256, 256, 1]), 800, "x–z plane, chunks 256 × 256 × 1: expected 800");
    assert_eq!(chunks_read(&plane, &[64, 64, 64]), 64, "x–z plane, chunks 64³: expected 64");
}
