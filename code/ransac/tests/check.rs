// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p ransac

use imgcourse::Rng;
use ransac::sim::line_points;
use ransac::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

const B1: [[f32; 2]; 8] = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.2], [5.0, 1.0], [1.0, 4.0], [4.0, 0.0]];

#[test]
fn c1_sample_1_b1a() {
    let l = line_through(B1[0], B1[3]);
    assert!(close(l.theta.to_degrees(), 135.0, 0.01) && close(l.rho, 0.0, 1e-5), "B1(a): got θ = {}°, ρ = {}, expected 135°, 0", l.theta.to_degrees(), l.rho);
    let d: Vec<f32> = B1.iter().map(|&p| distance(&l, p)).collect();
    let expected = [0.0, 0.0, 0.0, 0.0, 0.14, 2.83, 2.12, 2.83];
    for k in 0..8 {
        assert!(close(d[k], expected[k], 0.005), "B1(a): distances {d:?}, expected {expected:?}");
    }
    assert_eq!(d.iter().filter(|&&v| v <= 0.5).count(), 5, "B1(a): 5 inliers");
}

#[test]
fn c1_sample_2_b1b() {
    let l = line_through(B1[5], B1[6]);
    assert!(close(l.theta.cos(), 0.6, 1e-4) && close(l.theta.sin(), 0.8, 1e-4) && close(l.rho, 3.8, 1e-4), "B1(b): normal (0.6, 0.8), ρ = 3.8; got θ = {}°, ρ = {}", l.theta.to_degrees(), l.rho);
    let d: Vec<f32> = B1.iter().map(|&p| distance(&l, p)).collect();
    let expected = [3.8, 2.4, 1.0, 0.4, 1.96, 0.0, 0.0, 1.4];
    for k in 0..8 {
        assert!(close(d[k], expected[k], 0.005), "B1(b): distances {d:?}, expected {expected:?}");
    }
    // the same line, the points in the other order, gives the same θ and ρ
    let m = line_through(B1[6], B1[5]);
    assert!(close(m.theta, l.theta, 1e-5) && close(m.rho, l.rho, 1e-4), "the order of the two points must not matter");
}

#[test]
fn c2_iterations_b2() {
    assert_eq!(iterations_needed(0.5, 2, 0.99), 17, "B2(a)");
    assert_eq!(iterations_needed(0.5, 3, 0.99), 35, "B2(b)");
    assert_eq!(iterations_needed(0.2, 2, 0.99), 113, "B2(c)");
    assert_eq!(iterations_needed(0.3, 3, 0.99), 169, "B2(d)");
}

#[test]
fn c3_refit_b6() {
    let l = fit_line_tls(&B1[..5]);
    // the line passes through the centroid (2, 2.04); its direction is at 46.1°, the normal at 136.1°
    assert!(close(l.theta.to_degrees(), 136.1, 0.1), "B6: normal at 136.1°, got {}°", l.theta.to_degrees());
    assert!(close(distance(&l, [2.0, 2.04]), 0.0, 1e-4), "B6: the line must pass through the centroid (2, 2.04)");
    // a vertical line x = 3: θ = 0
    let v = fit_line_tls(&[[3.0, 0.0], [3.0, 1.0], [3.0, 5.0]]);
    assert!(close(v.theta.sin().abs(), 0.0, 1e-4) && close(distance(&v, [3.0, 9.0]), 0.0, 1e-4), "vertical line x = 3: got θ = {}°, ρ = {}", v.theta.to_degrees(), v.rho);
}

#[test]
fn c4_ransac_b1c() {
    let mut rng = Rng::new(1);
    let (l, inliers) = ransac_line(&B1, 0.5, 40, &mut rng);
    assert_eq!(inliers, vec![0, 1, 2, 3, 4], "B1(c): the inliers are the first five points");
    assert!(close(l.theta.to_degrees(), 136.1, 0.1), "B1(c): the refit is the TLS line at 136.1°, got {}°", l.theta.to_degrees());
}

#[test]
fn c4_ransac_with_outliers() {
    let mut rng = Rng::new(3);
    let pts = line_points(100.0, 63.0, 58.0, 0.8, 30, 30, &mut rng);
    let (l, inliers) = ransac_line(&pts, 2.0, 50, &mut rng);
    assert!(close(l.theta.to_degrees(), 63.0, 2.0) && close(l.rho, 58.0, 1.5), "50 % outliers: got θ = {}°, ρ = {}, expected 63°, 58", l.theta.to_degrees(), l.rho);
    assert!(inliers.len() >= 25, "most of the 30 line points should be inliers, got {}", inliers.len());
}

#[test]
fn c5_circle_b7() {
    let (a, b, r) = fit_circle(&[[5.0, 0.0], [0.0, 5.0], [-5.0, 0.0], [0.0, -5.0]]).expect("a circle");
    assert!(close(a, 0.0, 1e-4) && close(b, 0.0, 1e-4) && close(r, 5.0, 1e-4), "B7: got ({a}, {b}), r = {r}, expected (0, 0), 5");
    let (a, b, r) = fit_circle(&[[52.0 + 28.0, 48.0], [52.0, 48.0 + 28.0], [52.0 - 28.0, 48.0]]).expect("three points");
    assert!(close(a, 52.0, 1e-3) && close(b, 48.0, 1e-3) && close(r, 28.0, 1e-3), "three points of the circle (52, 48), r = 28: got ({a}, {b}), {r}");
    assert!(fit_circle(&[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]).is_none(), "three points on a line have no circle");
}
