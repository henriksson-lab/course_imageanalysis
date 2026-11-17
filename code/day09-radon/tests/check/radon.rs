// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day09 radon::

use imgcourse::Image;
use day09::radon::tomo::{angles, centre};
use day09::radon::*;
use std::f32::consts::PI;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_three_points_b2() {
    // A = (2, 0), B = (0, 1), C = (−1, −1) at 0°, 45°, 90°, 135°
    let expected = [[2.0, 0.0, -1.0], [1.41, 0.71, -1.41], [0.0, 1.0, -1.0], [-1.41, 0.71, 0.0]];
    let points = [(2.0, 0.0), (0.0, 1.0), (-1.0, -1.0)];
    for (i, deg) in [0.0f32, 45.0, 90.0, 135.0].iter().enumerate() {
        for (j, &(x, y)) in points.iter().enumerate() {
            let r = rho(x, y, deg.to_radians());
            assert!(close(r, expected[i][j], 0.01), "B2, point {} at {deg}°: ρ = {r:.2}, expected {}", ["A", "B", "C"][j], expected[i][j]);
        }
    }
}

#[test]
fn c1_crossing_b3() {
    // A and B lie on the same ray at θ = 63.4° (tan θ = 2), ρ = 0.89
    let t = 2.0f32.atan();
    let (a, b) = (rho(2.0, 0.0, t), rho(0.0, 1.0, t));
    assert!(close(a, b, 1e-5) && close(a, 0.894, 1e-3), "B3: at 63.4° A and B should both be at ρ = 0.89, got {a:.3} and {b:.3}");
}

/// The image of B7 (top row first): rows sums (1, 2, 1), column sums (1, 2, 1).
fn b7() -> Image<f32> {
    Image::from_rows(&[[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
}

#[test]
fn c2_column_sums_at_0() {
    let p = projection(&b7(), 0.0);
    // θ = 0: the rays run vertically; bin k is column k (ρ = x grows to the right)
    for (k, e) in [1.0, 2.0, 1.0].into_iter().enumerate() {
        assert!(close(p[k], e, 1e-4), "B7, projection at 0°, bin {k}: got {}, expected the column sum {e}", p[k]);
    }
}

#[test]
fn c2_row_sums_at_90() {
    let img = Image::from_rows(&[[1.0, 0.0, 0.0], [0.0, 3.0, 1.0], [0.0, 1.0, 0.0]]);
    let p = projection(&img, PI / 2.0);
    // θ = 90°: horizontal rays; ρ = y grows upwards, so bin 0 is the bottom row
    for (k, e) in [1.0, 4.0, 1.0].into_iter().enumerate() {
        assert!(close(p[k], e, 1e-4), "projection at 90°, bin {k}: got {}, expected {e} (bin 0 = bottom row)", p[k]);
    }
}

#[test]
fn c2_disc_b6() {
    // a disc of radius 20 pixels, value 1: the projection is the chord 2 √(400 − ρ²) at every angle
    let n = 64;
    let c = centre(n);
    let disc = Image::from_fn(n, n, |x, y| if (x as f32 - c).powi(2) + (y as f32 - c).powi(2) <= 400.0 { 1.0 } else { 0.0 });
    for deg in [0.0f32, 30.0, 90.0] {
        let p = projection(&disc, deg.to_radians());
        let middle = p[32]; // ρ = 0.5
        let chord = 2.0 * (400.0f32 - 0.25).sqrt();
        assert!(close(middle, chord, 1.5), "B6, disc at {deg}°, ρ = 0.5: got {middle:.1}, expected about {chord:.1}");
    }
}

#[test]
fn c3_sinogram_mass_and_symmetry() {
    let n = 32;
    let img = Image::from_fn(n, n, |x, y| {
        let (dx, dy) = (x as f32 - 12.0, y as f32 - 17.0);
        (-(dx * dx + dy * dy) / 18.0).exp()
    });
    let mut thetas = angles(12);
    thetas.extend(angles(12).iter().map(|t| t + PI)); // a full turn
    let sino = sinogram(&img, &thetas);
    assert_eq!((sino.width(), sino.height()), (24, n), "one column per angle, one row per bin");
    let total = img.sum();
    for a in 0..24 {
        let s: f32 = (0..n).map(|k| sino[(a, k)]).sum();
        assert!(close(s, total, 0.01 * total), "every projection sums to the total {total:.2}; column {a} sums to {s:.2}");
    }
    // p at θ + 180° is p at θ, mirrored: bin k ↔ bin n − 1 − k
    for a in 0..12 {
        for k in 0..n {
            let (p, q) = (sino[(a, k)], sino[(a + 12, n - 1 - k)]);
            assert!(close(p, q, 1e-3), "B5: angle {a}, bin {k}: {p:.4}, but 180° later the mirrored bin has {q:.4}");
        }
    }
}
