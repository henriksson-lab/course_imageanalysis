// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day37 agreement::

use day37::agreement::*;
use imgcourse::Image;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

const A: [(f32, f32); 3] = [(0., 0.), (1., 0.), (2., 0.)];
const B: [(f32, f32); 4] = [(0., 1.), (1., 1.), (2., 1.), (1., 4.)];

#[test]
fn c1_boundary_of_a_square() {
    // A 4 × 4 square: 12 boundary pixels, the 4 inner ones are not.
    let mask = Image::from_fn(8, 8, |x, y| (2..6).contains(&x) && (2..6).contains(&y));
    let b = boundary(&mask);
    assert_eq!(b.len(), 12, "a 4 × 4 square has 12 boundary pixels, got {}: {b:?}", b.len());
    assert!(!b.contains(&(3.0, 3.0)), "(3, 3) is inside the square, not on its boundary");
    // touching the image border counts as boundary
    let full = Image::new(3, 3, true);
    assert_eq!(boundary(&full).len(), 8, "a full 3 × 3 image: the 8 outer pixels are boundary");
}

#[test]
fn c2_nearest_distances_b1a() {
    let d = nearest_distances(&A, &B);
    assert!(d.iter().all(|&v| close(v, 1.0, 1e-5)), "B1a, from A: got {d:?}, expected [1, 1, 1]");
    let d = nearest_distances(&B, &A);
    let e = [1.0, 1.0, 1.0, 4.0];
    assert!(d.iter().zip(e).all(|(a, b)| close(*a, b, 1e-5)), "B1a, from B: got {d:?}, expected {e:?}");
}

#[test]
fn c3_boundary_scores_b1b() {
    let s = boundary_scores(&A, &B);
    assert!(close(s.hausdorff, 4.0, 1e-5), "B1b, Hausdorff: got {}, expected 4", s.hausdorff);
    assert!(close(s.asd, 10.0 / 7.0, 1e-4), "B1b, ASD: got {}, expected 10/7 = 1.43", s.asd);
    let s = boundary_scores(&A, &B[0..3]);
    assert!(close(s.asd, 1.0, 1e-5) && close(s.hausdorff, 1.0, 1e-5), "B1c, without (1, 4): got {s:?}, expected ASD 1 and Hausdorff 1");
}

#[test]
fn c3_hd95_ignores_a_far_point() {
    // 100 points on a line, the same line shifted by 1, plus one point 30 away.
    let a: Vec<(f32, f32)> = (0..100).map(|i| (i as f32, 0.0)).collect();
    let mut b: Vec<(f32, f32)> = (0..100).map(|i| (i as f32, 1.0)).collect();
    b.push((50.0, 31.0));
    let s = boundary_scores(&a, &b);
    assert!(close(s.hausdorff, 31.0, 1e-4), "Hausdorff: got {}, expected 31", s.hausdorff);
    assert!(close(s.hd95, 1.0, 1e-5), "HD95: got {}, expected 1 (one far point of 101 is less than 5 %)", s.hd95);
}

const MANUAL: [f32; 6] = [20., 35., 50., 65., 80., 100.];
const AUTO: [f32; 6] = [19., 33., 46., 60., 72., 88.];

#[test]
fn c4_bland_altman_b2() {
    let ba = bland_altman(&MANUAL, &AUTO);
    assert!(close(ba.bias, -5.333, 1e-3), "B2b, bias: got {}, expected -5.33", ba.bias);
    assert!(close(ba.sd, 4.08, 5e-3), "B2b, SD: got {}, expected 4.08", ba.sd);
    assert!(close(ba.lower, -13.3, 0.05) && close(ba.upper, 2.7, 0.05), "B2b, limits: got {} to {}, expected -13.3 to 2.7", ba.lower, ba.upper);
}

#[test]
fn c5_correlation_b2d() {
    let r = correlation(&MANUAL, &AUTO);
    assert!(close(r, 0.9995, 5e-4), "B2d: got r = {r}, expected 0.9995");
    // 80 % of the manual count: r = 1, although every count is wrong.
    let scaled: Vec<f32> = MANUAL.iter().map(|m| 0.8 * m).collect();
    assert!(close(correlation(&MANUAL, &scaled), 1.0, 1e-5), "auto = 0.8 × manual should give r = 1");
}
