// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p levelsets levelset_eq::

use imgcourse::Image;
use levelsets::levelset_eq::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// The 1D grid of the exercise, x = −3..=3, φ = |x| − 1, as a 7 × 1 image.
fn v() -> Image<f32> {
    Image::from_rows(&[[2.0, 1.0, 0.0, -1.0, 0.0, 1.0, 2.0]])
}

fn row(img: &Image<f32>) -> Vec<f32> {
    img.row(0).to_vec()
}

fn check_row(got: &[f32], expected: &[f32], name: &str) {
    for (g, e) in got.iter().zip(expected) {
        assert!(close(*g, *e, 2e-3), "{name}: got {got:?}, expected {expected:?}");
    }
}

#[test]
fn c1_upwind_slopes_b1() {
    let phi = v();
    let slopes: Vec<f32> = (0..7).map(|x| upwind_gradient(&phi, x, 0, true)).collect();
    check_row(&slopes, &[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0], "B1(a), outward upwind slopes");
    let inward = upwind_gradient(&phi, 3, 0, false);
    assert!(close(inward, 2f32.sqrt(), 1e-5), "B3(a), inward at x = 0: got {inward}, expected √2");
}

#[test]
fn c2_one_step_outward_b1() {
    let new = level_set_step(&v(), &Image::new(7, 1, 1.0), 0.5);
    check_row(&row(&new), &[1.5, 0.5, -0.5, -1.0, -0.5, 0.5, 1.5], "B1(b), F = 1, Δt = 0.5");
}

#[test]
fn c2_too_large_b2() {
    let new = level_set_step(&v(), &Image::new(7, 1, 1.0), 2.0);
    check_row(&row(&new), &[0.0, -1.0, -2.0, -1.0, -2.0, -1.0, 0.0], "B2, F = 1, Δt = 2");
}

#[test]
fn c2_shrinking_b3() {
    let speed = Image::new(7, 1, -1.0);
    let once = level_set_step(&v(), &speed, 0.5);
    assert!(close(once[(3, 0)], -0.293, 2e-3), "B3(a), x = 0: got {}, expected −0.29", once[(3, 0)]);
    assert!(close(once[(2, 0)], 0.5, 1e-5) && close(once[(4, 0)], 0.5, 1e-5), "B3(a), x = ±1: got {} and {}, expected 0.5", once[(2, 0)], once[(4, 0)]);
    let twice = level_set_step(&once, &speed, 0.5);
    assert!(close(twice[(3, 0)], 0.27, 5e-3), "B3(b), x = 0 after two steps: got {}, expected 0.27", twice[(3, 0)]);
    assert!(twice.pixels().iter().all(|&p| p > 0.0), "B3(b): after two steps everything should be outside: {:?}", row(&twice));
}

#[test]
fn c3_curvature_flow_of_a_circle_b5() {
    // Radius 20, ε = 1: r² = 400 − 2t, so at t = 100 the radius is √200 = 14.1 (area 628).
    let mut phi = Image::from_fn(64, 64, |x, y| ((x as f32 - 32.0).powi(2) + (y as f32 - 32.0).powi(2)).sqrt() - 20.0);
    for _ in 0..400 {
        phi = curvature_step(&phi, 1.0, 0.25);
    }
    let area = phi.pixels().iter().filter(|&&v| v < 0.0).count() as f32;
    let expected = std::f32::consts::PI * 200.0;
    assert!((area - expected).abs() < 0.05 * expected, "after t = 100: {area} pixels inside, expected about {expected:.0}");
}

#[test]
fn c4_reinitialise_a_scaled_distance() {
    let d = |x: usize, y: usize| ((x as f32 - 20.0).powi(2) + (y as f32 - 20.0).powi(2)).sqrt() - 10.0;
    let phi = Image::from_fn(40, 40, |x, y| 3.0 * d(x, y));
    let r = reinitialise(&phi);
    // Distances to the crossing points, not to the curve itself: up to about 0.2 too large.
    for (x, y) in r.coords() {
        if d(x, y).abs() < 5.0 {
            assert!(close(r[(x, y)], d(x, y), 0.25), "pixel ({x}, {y}): got {}, expected the signed distance {}", r[(x, y)], d(x, y));
        }
    }
}
