// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day45 acceptor_bleach::

use day45::acceptor_bleach::sim::{self, Bleach};
use day45::acceptor_bleach::*;
use imgcourse::Image;

fn close(got: f32, expected: f32, tol: f32) -> bool {
    (got - expected).abs() < tol
}

#[test]
fn c1_efficiency_b1() {
    let pre = Image::from_rows(&[[600.0, 600.0, 50.0], [600.0, 0.0, 10.0]]);
    let post = Image::from_rows(&[[800.0, 800.0, 50.0], [800.0, 0.0, 20.0]]);
    let e = efficiency_map(&pre, &post);
    assert!(close(e[(0, 0)], 0.25, 1e-6), "B1: got E = {}, expected 0.25", e[(0, 0)]);
    assert_eq!(e[(1, 1)], 0.0, "post = 0 should give 0, got {}", e[(1, 1)]);
    let box_ = Image::from_rows(&[[true, true, false], [true, false, false]]);
    let r = region_efficiency(&pre, &post, &box_);
    assert!(close(r, 0.25, 1e-6), "B1, region: got {r}, expected 0.25");
    let all = Image::new(3, 2, true);
    let r = region_efficiency(&pre, &post, &all);
    assert!(close(r, 1.0 - 1860.0 / 2470.0, 1e-6), "ratio of sums over all pixels: got {r}, expected 1 − 1860/2470");
}

#[test]
fn c2_incomplete_bleaching_b2() {
    let e = correct_incomplete(0.25, 0.8);
    assert!(close(e, 0.294, 1e-3), "B2(b): got {e}, expected 0.294");
    // B2(c): back to the observed value
    let obs = 0.8 * e / (1.0 - 0.2 * e);
    assert!(close(obs, 0.25, 1e-5), "B2(c): with the corrected E and b = 0.8 the observed E should be 0.25, got {obs}");
    assert!(close(correct_incomplete(0.3, 1.0), 0.3, 1e-6), "b = 1: nothing to correct");
}

#[test]
fn c2_donor_bleaching_b3() {
    let pre = Image::from_rows(&[[500.0, 600.0]]);
    let post = Image::from_rows(&[[475.0, 800.0]]);
    let control = Image::from_rows(&[[true, false]]);
    let ratio = control_ratio(&pre, &post, &control);
    assert!(close(ratio, 0.95, 1e-6), "B3: control ratio got {ratio}, expected 0.95");
    let fixed = post.map(|v| v / ratio);
    let bleached = Image::from_rows(&[[false, true]]);
    let e = region_efficiency(&pre, &fixed, &bleached);
    assert!(close(e, 0.2875, 1e-4), "B3(a): got {e}, expected 0.2875");
    let both = correct_incomplete(e, 0.8);
    assert!(close(both, 0.335, 1e-3), "B3(b): got {both}, expected 0.335");
}

#[test]
fn c3_shift_and_find_shift() {
    let img = Image::from_rows(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let s = shift(&img, 1, 0);
    assert_eq!(s.row(0), &[2.0, 3.0, 0.0], "shift by 1 in x: row 0 should be [2, 3, 0]");
    let s = shift(&img, 0, -1);
    assert_eq!(s.row(1), &[1.0, 2.0, 3.0], "shift by −1 in y: row 1 should be the old row 0");
    let ex = sim::experiment(&Bleach { drift: (3, -2), ..Bleach::default() });
    let (dx, dy) = find_shift(&ex.donor_pre, &ex.donor_post, 5);
    assert_eq!((dx, dy), (3, -2), "the cells moved by (3, −2) between before and after; got ({dx}, {dy})");
}

#[test]
fn c4_sigma_e_b4() {
    for (p, q, expected) in [(600.0, 800.0, 0.0405), (60.0, 80.0, 0.128), (6000.0, 8000.0, 0.0128)] {
        let s = sigma_e(p, q);
        assert!(close(s, expected, 5e-4), "B4, {p}/{q} photons: got σ_E = {s:.4}, expected {expected}");
    }
}
