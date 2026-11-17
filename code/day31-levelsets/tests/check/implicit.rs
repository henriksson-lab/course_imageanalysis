// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day31 implicit::

use imgcourse::Image;
use day31::implicit::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// `f(x, y)` sampled at x, y = −2..=2: pixel (i, j) is the point (i − 2, j − 2).
fn grid5(f: impl Fn(f32, f32) -> f32) -> Image<f32> {
    Image::from_fn(5, 5, |i, j| f(i as f32 - 2.0, j as f32 - 2.0))
}

/// The 5 × 5 grid of the exercise: φ = √(x² + y²) − 1.5.
fn table() -> Image<f32> {
    grid5(|x, y| (x * x + y * y).sqrt() - 1.5)
}

#[test]
fn c1_ring_b4() {
    // Ring between radius 1 and 2 on a grid of spacing 0.5, centre at pixel (6, 6).
    let disk = |r: f32| Image::from_fn(13, 13, move |i, j| ((i as f32 - 6.0).powi(2) + (j as f32 - 6.0).powi(2)).sqrt() * 0.5 - r);
    let ring = difference(&disk(2.0), &disk(1.0));
    for ((i, j), expected, name) in [((6, 6), 1.0, "the origin"), ((9, 6), -0.5, "ρ = 1.5"), ((12, 6), 1.0, "ρ = 3")] {
        assert!(close(ring[(i, j)], expected, 1e-5), "B4(b), {name}: got {}, expected {expected}", ring[(i, j)]);
    }
    let ring2 = intersection(&disk(2.0), &disk(1.0).map(|v| -v));
    assert_eq!(ring, ring2, "difference(A, B) should equal intersection(A, complement of B)");
    let both = union(&disk(1.0), &disk(2.0));
    assert_eq!(both, disk(2.0), "the union of the two disks is the big disk");
}

#[test]
fn c2_crossings_b1() {
    let phi = table();
    let c = row_crossings(&phi, 2); // the row y = 0
    assert!(c.len() == 2 && close(c[0] - 2.0, -1.5, 1e-4) && close(c[1] - 2.0, 1.5, 1e-4), "B1(b), row y = 0: got x = {:?}, expected ±1.5", c.iter().map(|v| v - 2.0).collect::<Vec<_>>());
    let c = row_crossings(&phi, 3); // the row y = 1
    assert!(c.len() == 2 && close(c[1] - 2.0, 1.104, 2e-3) && close(c[0] - 2.0, -1.104, 2e-3), "B1(c), row y = 1: got x = {:?}, expected ±1.104", c.iter().map(|v| v - 2.0).collect::<Vec<_>>());
    let c = row_crossings(&phi, 0); // the row y = −2: all outside
    assert!(c.is_empty(), "row y = −2 is outside the circle: got crossings {c:?}");
}

#[test]
fn c3_normals_b2_b3() {
    let phi = table();
    let (n, len) = normal(&phi, 3, 2); // the point (1, 0)
    assert!(close(n[0], 1.0, 1e-5) && close(n[1], 0.0, 1e-5) && close(len, 1.0, 1e-4), "B2(a), at (1, 0): normal {n:?}, |∇φ| = {len}, expected (1, 0) and 1");
    let (n, len) = normal(&phi, 3, 3); // the point (1, 1)
    let r = std::f32::consts::FRAC_1_SQRT_2;
    assert!(close(n[0], r, 1e-4) && close(n[1], r, 1e-4), "B3, at (1, 1): normal {n:?}, expected (0.707, 0.707)");
    assert!(close(len, 0.874, 2e-3), "B3, at (1, 1): |∇φ| = {len}, expected 0.874");
}

#[test]
fn c4_curvature_b2() {
    let k = curvature(&table());
    assert!(close(k[(3, 2)], 0.828, 2e-3), "B2(b), at (1, 0): κ = {}, expected 0.828", k[(3, 2)]);
}

#[test]
fn c4_curvature_of_a_large_circle() {
    // A circle of radius 20 pixels: κ = 1/20 on it, the same for φ = distance, 3 × distance and
    // x² + y² − r² (the curvature does not depend on how φ is scaled).
    let c = 32.0;
    let d = |i: usize, j: usize| ((i as f32 - c).powi(2) + (j as f32 - c).powi(2)).sqrt();
    let phis = [
        Image::from_fn(64, 64, |i, j| d(i, j) - 20.0),
        Image::from_fn(64, 64, |i, j| 3.0 * (d(i, j) - 20.0)),
        Image::from_fn(64, 64, |i, j| d(i, j).powi(2) - 400.0),
    ];
    for (k, phi) in phis.iter().enumerate() {
        let kappa = curvature(phi);
        for (i, j) in [(52, 32), (32, 12), (46, 46)] {
            let expected = 1.0 / d(i, j);
            assert!(close(kappa[(i, j)], expected, 2e-3), "φ number {k}, pixel ({i}, {j}): κ = {}, expected {expected}", kappa[(i, j)]);
        }
    }
}
