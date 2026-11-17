// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p colocalisation coloc_confounds::

use colocalisation::coloc_confounds::sim;
use colocalisation::common::{self, Scene};
use colocalisation::coloc_confounds::*;
use imgcourse::Image;

fn close(got: f32, expected: f32, tol: f32) -> bool {
    (got - expected).abs() < tol
}

#[test]
fn c1_leak_slope_and_compensation_b1() {
    let a_c = Image::from_rows(&[[0.0, 10.0, 5.0, 20.0]]);
    let b_c = Image::from_rows(&[[0.1, 3.1, 1.4, 6.0]]);
    let s = leak_slope(&a_c, &b_c, &common::everywhere(4, 1));
    assert!(close(s, 158.0 / 525.0, 1e-4), "B1(c): got ŝ = {s}, expected 0.301");
    let all = common::everywhere(6, 1);
    let a = Image::from_rows(&[[0.0, 10.0, 2.0, 8.0, 0.0, 4.0]]);
    let b = Image::from_rows(&[[3.0, 4.0, 1.0, 2.0, 5.0, 3.0]]);
    let leaked = b.zip_map(&a, |vb, va| vb + 0.3 * va);
    let r = common::pearson(&a, &leaked, &all);
    assert!(close(r, 0.64, 0.01), "B1(b): r with 30 % leak should be 0.64, got {r} (is the check itself right?)");
    let fixed = compensate(&leaked, &a, s);
    assert!(close(fixed[(1, 0)], 3.99, 0.01), "B1(d): compensated B at the second pixel: got {}, expected 3.99", fixed[(1, 0)]);
    let r = common::pearson(&a, &fixed, &all);
    assert!(close(r, -0.07, 0.01), "B1(d): r after compensation: got {r}, expected −0.07");
    let over = compensate(&leaked, &a, 0.6);
    let r = common::pearson(&a, &over, &all);
    assert!(close(r, -0.69, 0.01), "B1(e): r after over-compensation with 0.6: got {r}, expected −0.69");
}

#[test]
fn c2_gaussian_overlap_b2_b3() {
    let checks = [
        (78.0, 78.0, 100.0, 0.66, "B2(b), 100 nm"),
        (78.0, 78.0, 300.0, 0.025, "B2(b), 300 nm"),
        (78.0, 78.0, 130.0, 0.50, "B2(c), 130 nm"),
        (80.0, 80.0, 65.0, 0.85, "B3, one pixel"),
        (80.0, 80.0, 130.0, 0.52, "B3, two pixels"),
        (78.0, 100.0, 0.0, 0.97, "same place, widths 78 and 100 nm"),
    ];
    for (s1, s2, d, expected, what) in checks {
        let g = gaussian_overlap(s1, s2, d);
        assert!(close(g, expected, 0.006), "{what}: got {g:.3}, expected {expected}");
    }
}

#[test]
fn c3_top_hat_keeps_spots_removes_ramps() {
    // a ramp (uneven background) plus one bright pixel in the middle
    let mut img = Image::from_fn(21, 21, |x, _| 0.2 + 0.01 * x as f32);
    img[(10, 10)] += 1.0;
    let out = subtract_background(&img, 3);
    assert!(close(out[(10, 10)], 1.0, 1e-4), "the spot should keep its height 1 above the ramp, got {}", out[(10, 10)]);
    for (x, y) in [(5, 5), (15, 8), (8, 15)] {
        assert!(close(out[(x, y)], 0.0, 1e-4), "pixel ({x}, {y}) of the ramp should become 0, got {}", out[(x, y)]);
    }
    let flat = subtract_background(&Image::new(9, 9, 0.3), 2);
    assert!(flat.pixels().iter().all(|&v| v.abs() < 1e-6), "a constant offset should become 0 everywhere");
}

#[test]
fn c4_estimate_shift_on_beads() {
    let p = Scene::default();
    for shift in [(1.5, 0.0), (-0.7, 1.2)] {
        let (a, b) = sim::beads(&p, 25, shift);
        let (tx, ty) = estimate_shift(&a, &b, 4);
        assert!(close(tx, shift.0, 0.15) && close(ty, shift.1, 0.15), "beads with B moved by {shift:?}: got ({tx:.2}, {ty:.2})");
    }
}

#[test]
fn c4_shift_image() {
    let img = Image::from_rows(&[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]);
    let s = shift_image(&img, 1.0, 0.0);
    assert!(close(s[(0, 0)], 1.0, 1e-6) && close(s[(3, 1)], 7.0, 1e-6), "shift by 1 in x: got row 0 = {:?}", s.row(0));
    let s = shift_image(&img, 0.5, 0.5);
    assert!(close(s[(0, 0)], 2.5, 1e-6), "half a pixel in x and y: the mean of 0, 1, 4, 5 is 2.5, got {}", s[(0, 0)]);
    // a shift and the opposite shift give back the image (away from the edges)
    let (a, _) = sim::beads(&Scene::default(), 10, (0.0, 0.0));
    let back = shift_image(&shift_image(&a, 2.0, -1.0), -2.0, 1.0);
    assert!(close(back[(60, 60)], a[(60, 60)], 1e-6), "integer shift there and back: got {}, expected {}", back[(60, 60)], a[(60, 60)]);
}
