// Checks for the coding exercises, using the numbers of the pen-and-paper exercises (B1, B2, B4)
// and frame pairs with a known motion.
// Run: cargo test -p opticalflow horn_schunck::

use opticalflow::horn_schunck::*;
use imgcourse::Image;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_hs_update_b1_b2() {
    for (lambda, expected) in [(0.0, 2.0), (0.25, 1.6), (1.0, 1.0), (3.0, 0.5)] {
        let (u, v) = hs_update(2.0, 0.0, -4.0, 0.0, 0.0, lambda);
        assert!(close(u, expected, 1e-5) && close(v, 0.0, 1e-6), "B1(a), λ = {lambda}: got ({u}, {v}), expected ({expected}, 0)");
    }
    let (u, v) = hs_update(2.0, 0.0, -4.0, 2.0, 1.0, 1.0);
    assert!(close(u, 2.0, 1e-6) && close(v, 1.0, 1e-6), "B1(b): the average is on the line, got ({u}, {v}), expected (2, 1)");
    let (u, v) = hs_update(1.0, 1.0, -2.0, 0.0, 0.0, 1.0);
    assert!(close(u, 1.0 / 3.0, 1e-5) && close(v, 1.0 / 3.0, 1e-5), "B2(a): got ({u}, {v}), expected (1/3, 1/3)");
    let (u, v) = hs_update(1.0, 1.0, -2.0, 1.0, 0.0, 1.0);
    assert!(close(u, 7.0 / 6.0, 1e-5) && close(v, 1.0 / 6.0, 1e-5), "B2(b): got ({u}, {v}), expected (1.17, 0.17)");
}

#[test]
fn c2_neighbour_average() {
    let f = Image::from_rows(&[[0.0, 1.0, 0.0], [2.0, 9.0, 4.0], [0.0, 3.0, 0.0]]);
    assert!(close(neighbour_average(&f, 1, 1), 2.5, 1e-6), "centre: (1 + 2 + 4 + 3)/4 = 2.5, got {}", neighbour_average(&f, 1, 1));
    // Corner (0, 0): left and up are outside, the edge is copied: (0 + 1 + 0 + 2)/4.
    assert!(close(neighbour_average(&f, 0, 0), 0.75, 1e-6), "corner: got {}, expected 0.75", neighbour_average(&f, 0, 0));
}

/// A smooth textured image moved by (dx, dy).
fn pattern(size: usize, dx: f32, dy: f32) -> Image<f32> {
    Image::from_fn(size, size, |x, y| {
        let (x, y) = (x as f32 - dx, y as f32 - dy);
        0.5 + 0.15 * (x / 4.0).sin() + 0.15 * (y / 5.0 + 0.3 * x / 7.0).cos()
    })
}

#[test]
fn c3_hs_sweep_one_step() {
    // Zero flow and one sweep: every pixel does the update of C1 from the average 0.
    let (ix, iy, it) = (Image::new(3, 3, 2.0f32), Image::new(3, 3, 0.0f32), Image::new(3, 3, -4.0f32));
    let zero = Image::new(3, 3, 0.0f32);
    let (u, v) = hs_sweep(&ix, &iy, &it, &zero, &zero, 1.0);
    assert!(close(u[(1, 1)], 1.0, 1e-6) && close(v[(1, 1)], 0.0, 1e-6), "one sweep from zero, λ = 1 (B1): got ({}, {}), expected (1, 0)", u[(1, 1)], v[(1, 1)]);
    // The second sweep starts from the average of the first: 1, so r = (2 − 4)/8 and u = 1.5.
    let (u, _) = hs_sweep(&ix, &iy, &it, &u, &v, 1.0);
    assert!(close(u[(1, 1)], 1.5, 1e-6), "second sweep: got {}, expected 1.5", u[(1, 1)]);
}

#[test]
fn c3_horn_schunck_finds_a_shift() {
    let (f1, f2) = (pattern(40, 0.0, 0.0), pattern(40, 0.5, -0.3));
    let (ix, iy, it) = gradients(&f1, &f2);
    let (u, v) = horn_schunck(&ix, &iy, &it, 0.001, 300);
    let (a, b) = (u[(20, 20)], v[(20, 20)]);
    assert!(close(a, 0.5, 0.05) && close(b, -0.3, 0.05), "a pattern moved by (0.5, −0.3): got ({a:.3}, {b:.3}) at the centre");
}

#[test]
fn c3_filling_in() {
    // A textured frame with a flat 15 × 15 square in the middle, all moved by 1 px to the right.
    let tex = |dx: f32| {
        let p = pattern(41, dx, 0.0);
        p.map_xy(|x, y, v| if (13.0..28.0).contains(&(x as f32 - dx)) && (13..28).contains(&y) { 0.5 } else { v })
    };
    let (ix, iy, it) = gradients(&tex(0.0), &tex(1.0));
    let (u5, _) = horn_schunck(&ix, &iy, &it, 0.001, 5);
    let (u, _) = horn_schunck(&ix, &iy, &it, 0.001, 2000);
    assert!(u5[(20, 20)].abs() < 0.05, "after 5 sweeps the flow has not reached the centre of the flat square yet: got {}", u5[(20, 20)]);
    assert!(u[(20, 20)] > 0.6, "after 2000 sweeps it is filled in from the textured surroundings: got {}, expected near 1", u[(20, 20)]);
}

#[test]
fn c4_error_measures_b4() {
    let cases = [((2.0, 1.0), (2.5, 1.0), 5.8), ((0.1, 0.0), (0.6, 0.0), 25.3), ((5.0, 0.0), (5.5, 0.0), 1.0)];
    for (est, truth, ae) in cases {
        let e = endpoint_error(est, truth);
        let a = angular_error(est, truth);
        assert!(close(e, 0.5, 1e-5), "B4 {est:?} against {truth:?}: EPE {e}, expected 0.5");
        assert!(close(a, ae, 0.1), "B4 {est:?} against {truth:?}: AE {a:.2}°, expected {ae}°");
    }
    assert!(angular_error((1.0, 2.0), (1.0, 2.0)) < 0.05, "equal flows: AE 0");
}

#[test]
fn c5_warp_residual() {
    let (f1, f2) = (pattern(30, 0.0, 0.0), pattern(30, 2.0, 1.0));
    let (u, v) = (Image::new(30, 30, 2.0), Image::new(30, 30, 1.0));
    let r = warp_residual(&f1, &f2, &u, &v);
    assert!(r[(10, 10)].abs() < 1e-5 && r[(20, 15)].abs() < 1e-5, "the true flow leaves no residual: got {} and {}", r[(10, 10)], r[(20, 15)]);
    let zero = Image::new(30, 30, 0.0);
    let r0 = warp_residual(&f1, &f2, &zero, &zero);
    assert!(close(r0[(10, 10)], f2[(10, 10)] - f1[(10, 10)], 1e-6), "zero flow: the residual is f2 − f1");
}
