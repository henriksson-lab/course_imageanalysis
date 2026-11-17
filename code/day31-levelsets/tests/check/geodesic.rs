// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day31 geodesic::

use day31::geodesic::*;
use imgcourse::Image;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_edge_stopping_b1() {
    // B1(a): the step 0 0 0 10 10 10 smoothed with [1 2 1]/4 is 0 0 2.5 7.5 10 10.
    let smoothed = Image::from_rows(&[[0.0, 0.0, 2.5, 7.5, 10.0, 10.0]]);
    let g = edge_stopping(&smoothed, 2.0);
    let expected = [1.0, 0.719, 0.221, 0.221, 0.719, 1.0];
    for i in 0..6 {
        assert!(close(g[(i, 0)], expected[i], 2e-3), "B1(b), pixel {}: g = {}, expected {}", i + 1, g[(i, 0)], expected[i]);
    }
    let g = edge_stopping(&smoothed, 1.0);
    assert!(close(g[(2, 0)], 0.066, 2e-3), "B1(c), k = 1: g at the edge = {}, expected 0.066", g[(2, 0)]);
}

#[test]
fn c2_flat_region_is_a_plain_level_set_step() {
    // g = 1, no curvature: the balloon alone is the level set step of the previous unit (its B1).
    let phi = Image::from_rows(&[[2.0, 1.0, 0.0, -1.0, 0.0, 1.0, 2.0]]);
    let new = braking_step(&phi, &Image::new(7, 1, 1.0), 1.0, 0.0, 0.5);
    let expected = [1.5, 0.5, -0.5, -1.0, -0.5, 0.5, 1.5];
    for i in 0..7 {
        assert!(close(new[(i, 0)], expected[i], 1e-5), "g = 1, c = 1, Δt = 0.5: got {:?}, expected {expected:?}", new.row(0));
    }
    // On an edge with g = 0.1 the front moves a tenth as far.
    let new = braking_step(&phi, &Image::new(7, 1, 0.1), 1.0, 0.0, 0.5);
    assert!(close(new[(2, 0)], -0.05, 1e-5), "g = 0.1: φ at x = −1 became {}, expected −0.05", new[(2, 0)]);
}

#[test]
fn c2_balloon_against_curvature_b3() {
    // A circle of radius 8, g = 1: F = c − ε/r. With c = ε = 0.3 it grows (r* = 1); with
    // c = 0.05, ε = 1 it shrinks (r* = 20).
    let circle = Image::from_fn(40, 40, |x, y| ((x as f32 - 20.0).powi(2) + (y as f32 - 20.0).powi(2)).sqrt() - 8.0);
    let g = Image::new(40, 40, 1.0);
    let area = |p: &Image<f32>| p.pixels().iter().filter(|&&v| v < 0.0).count();
    let a0 = area(&circle);
    let (mut grow, mut shrink) = (circle.clone(), circle.clone());
    for _ in 0..40 {
        grow = braking_step(&grow, &g, 0.3, 0.3, 0.25);
        shrink = braking_step(&shrink, &g, 0.05, 1.0, 0.25);
    }
    assert!(area(&grow) > a0 + 20, "c = ε = 0.3: {} pixels inside after t = 10, expected more than at the start ({a0})", area(&grow));
    assert!(area(&shrink) + 20 < a0, "c = 0.05, ε = 1: {} pixels inside after t = 10, expected fewer than at the start ({a0})", area(&shrink));
}

#[test]
fn c3_advection_into_the_valley_b1d() {
    // g has a valley at the edge between pixels 3 and 4 (B1(b)): the zero crossing of φ, placed
    // left of it, must move right, into the valley.
    let g = Image::from_rows(&[[1.0, 1.0, 0.719, 0.221, 0.221, 0.719, 1.0, 1.0]]);
    let phi = Image::from_rows(&[[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]]); // crossing at 2.5
    let new = advection_step(&phi, &g, 1.0, 1.0);
    // At pixel 2: v = −(0.221 − 1)/2 = 0.39 > 0, backward difference 1: φ − 0.39.
    assert!(close(new[(2, 0)], -0.5 - 0.3895, 1e-3), "pixel 2: got {}, expected −0.89", new[(2, 0)]);
    let crossing = |p: &Image<f32>| (0..7).find_map(|i| (p[(i, 0)] < 0.0 && p[(i + 1, 0)] >= 0.0).then(|| i as f32 + p[(i, 0)] / (p[(i, 0)] - p[(i + 1, 0)]))).unwrap();
    assert!(crossing(&new) > 2.5, "the crossing should move right, towards the valley: from 2.5 to {}", crossing(&new));
}

#[test]
fn c3_linear_transport() {
    // φ = x − 3 and g = 0.1 x: v = −0.1 everywhere (to the left), so φ rises by 0.1 per unit time.
    let phi = Image::from_fn(7, 3, |x, _| x as f32 - 3.0);
    let g = Image::from_fn(7, 3, |x, _| 0.1 * x as f32);
    let new = advection_step(&phi, &g, 1.0, 1.0);
    for x in 1..6 {
        assert!(close(new[(x, 1)], phi[(x, 1)] + 0.1, 1e-5), "pixel ({x}, 1): got {}, expected {}", new[(x, 1)], phi[(x, 1)] + 0.1);
    }
}
