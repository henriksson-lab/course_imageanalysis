// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p pointdetection spot_fit::
// A 1D example of the exercise is a patch one pixel high; its middle sample is pixel x = 1
// (three samples) or x = 2 (five samples), so offsets are added to that.

use imgcourse::Image;
use pointdetection::spot_fit::sim::spot_mean;
use pointdetection::spot_fit::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

const B1: [f32; 3] = [0.430, 0.956, 0.783];

#[test]
fn c1_centroid_b1() {
    let c = centroid(&Image::from_rows(&[B1]), 0.0);
    assert!(close(c[0] - 1.0, 0.16, 0.005), "B1(a): centroid offset got {}, expected 0.16", c[0] - 1.0);
    assert!(close(c[1], 0.0, 1e-6), "one row: y should be 0, got {}", c[1]);
    // B1(d): a background of 0.5 that is not subtracted pulls it to the centre
    let c = centroid(&Image::from_rows(&[B1.map(|v| v + 0.5)]), 0.0);
    assert!(close(c[0] - 1.0, 0.10, 0.005), "B1(d): centroid offset with background got {}, expected 0.10", c[0] - 1.0);
    let c = centroid(&Image::from_rows(&[B1.map(|v| v + 0.5)]), 0.5);
    assert!(close(c[0] - 1.0, 0.16, 0.005), "background 0.5 subtracted again: got {}, expected 0.16", c[0] - 1.0);
}

#[test]
fn c1_centroid_bias_b9() {
    // B9: S = 100 at δ = 0.3 (70 on pixel 0, 30 on pixel 1), background 5 on pixels −3 … 3.
    let row = [5., 5., 5., 75., 35., 5., 5.];
    let c = centroid(&Image::from_rows(&[row]), 0.0);
    assert!(close(c[0] - 3.0, 30.0 / 135.0, 1e-4), "B9: got {}, expected 30/135 = 0.22", c[0] - 3.0);
}

#[test]
fn c2_three_point_b1() {
    let [m, c, p] = B1;
    assert!(close(parabola_vertex(m, c, p), 0.25, 0.005), "B1(b): parabola got {}, expected 0.25", parabola_vertex(m, c, p));
    assert!(close(log_parabola_vertex(m, c, p), 0.30, 0.005), "B1(c): log-parabola got {}, expected 0.30", log_parabola_vertex(m, c, p));
    let [m, c, p] = B1.map(|v| v + 0.5);
    assert!(close(parabola_vertex(m, c, p), 0.25, 0.005), "B1(d): parabola with background got {}, expected 0.25", parabola_vertex(m, c, p));
    assert!(close(log_parabola_vertex(m, c, p), 0.28, 0.005), "B1(d): log-parabola with background got {}, expected 0.28", log_parabola_vertex(m, c, p));
}

#[test]
fn c2_log_parabola_exact_in_2d() {
    let patch = spot_mean(13, &[(6.3, 5.8, 500.0)], 1.0, 5.0);
    let p = three_point(&patch, 5.0, true);
    assert!(close(p[0], 6.3, 1e-3) && close(p[1], 5.8, 1e-3), "log-parabola, background subtracted: got {p:?}, expected [6.3, 5.8]");
    let q = three_point(&patch, 5.0, false);
    assert!((q[0] - 6.3).abs() > 0.02, "the plain parabola should be biased (pixel locking), got {q:?}");
}

/// B7: a 1D spot with s = 1, samples at x = −2 … 2 (pixels 0 … 4), true x0 = 0.3, A = 30, B = 5.
fn b7() -> Image<f32> {
    Image::from_rows(&[[7.13, 17.89, 33.68, 28.48, 12.07]])
}

#[test]
fn c3_amplitude_background_b7a() {
    let (a, b) = fit_amplitude_background(&b7(), 2.3, 0.0, 1.0);
    assert!(close(a, 30.0, 0.05) && close(b, 5.0, 0.05), "B7(a): got A = {a}, B = {b}, expected 30 and 5");
}

#[test]
fn c4_gauss_newton_step_b7b() {
    let [dx, dy] = position_step(&b7(), 2.0, 0.0, 1.0, 30.0, 5.0);
    assert!(close(dx, 0.293, 0.002), "B7(b): one step from x0 = 0 got Δx0 = {dx}, expected 0.293");
    assert!(dy == 0.0, "a patch one pixel high cannot move in y, got {dy}");
}

#[test]
fn c5_fit_converges() {
    let f = fit_spot(&b7(), [2.0, 0.0], 1.0, 5);
    assert!(close(f.x - 2.0, 0.300, 0.002), "B7: the fit should end at 0.300, got {}", f.x - 2.0);
    let patch = spot_mean(13, &[(6.3, 5.8, 1000.0)], 1.0, 20.0);
    let f = fit_spot(&patch, [6.0, 6.0], 1.0, 8);
    assert!(close(f.x, 6.3, 1e-3) && close(f.y, 5.8, 1e-3), "2D spot at (6.3, 5.8): got ({}, {})", f.x, f.y);
    let peak = 1000.0 / (2.0 * std::f32::consts::PI);
    assert!(close(f.amplitude, peak, 0.5) && close(f.background, 20.0, 0.05), "amplitude {} (expected {peak}), background {} (expected 20)", f.amplitude, f.background);
}
