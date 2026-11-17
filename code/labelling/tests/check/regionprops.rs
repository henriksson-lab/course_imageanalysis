// Checks for the coding exercises, using the objects of the pen-and-paper exercises.
// Run: cargo test -p labelling regionprops::

use imgcourse::Image;
use labelling::regionprops::*;
use std::f32::consts::PI;

/// A mask written row by row: "1" or "#" = object, anything else = background.
fn mask(rows: &[&str]) -> Image<bool> {
    let cells: Vec<Vec<bool>> = rows.iter().map(|r| r.split_whitespace().map(|c| c == "1" || c == "#").collect()).collect();
    Image::from_fn(cells[0].len(), cells.len(), |x, y| cells[y][x])
}

/// The 12-pixel object of B1 as label 1, and the 2 × 10 rectangle of B5 (b) as label 2.
fn b1_and_b5() -> Image<u32> {
    let b1 = mask(&["1 1 0 0 0", "1 1 1 0 0", "0 1 1 1 0", "0 0 1 1 1", "0 0 0 1 0"]);
    Image::from_fn(20, 8, |x, y| {
        if x < 5 && y < 5 && b1[(x, y)] {
            1
        } else if (8..18).contains(&x) && (6..8).contains(&y) {
            2
        } else {
            0
        }
    })
}

/// A digital disc: the pixels whose centre lies within `r` of the centre pixel.
fn disc(r: f32) -> Image<bool> {
    let n = 2 * r.ceil() as usize + 5;
    let c = (n / 2) as f32;
    Image::from_fn(n, n, |x, y| (x as f32 - c).powi(2) + (y as f32 - c).powi(2) <= r * r)
}

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn c1_area_centroid_box_b1() {
    let regions = measure(&b1_and_b5());
    assert_eq!(regions.len(), 2, "two labels: got {} regions", regions.len());
    let r = &regions[0];
    assert_eq!(r.area, 12, "B1 (a): area");
    assert_eq!((r.x_min, r.x_max, r.y_min, r.y_max), (0, 4, 0, 4), "B1 (a): bounding box (x_min, x_max, y_min, y_max)");
    assert!(close(r.extent(), 0.48, 1e-4), "B1 (a): extent: got {}, expected 0.48", r.extent());
    assert!(close(r.cx, 22.0 / 12.0, 1e-4) && close(r.cy, 22.0 / 12.0, 1e-4), "B1 (b): centroid: got ({}, {}), expected (1.833, 1.833)", r.cx, r.cy);
}

#[test]
fn c1_central_moments_b1() {
    let r = &measure(&b1_and_b5())[0];
    assert!(close(r.mu20, 53.0 / 3.0, 1e-3), "B1 (c): μ20: got {}, expected 17.67", r.mu20);
    assert!(close(r.mu11, 41.0 / 3.0, 1e-3), "B1 (c): μ11: got {}, expected 13.67", r.mu11);
    assert!(close(r.mu02, 53.0 / 3.0, 1e-3), "B1 (c): μ02: got {}, expected 17.67", r.mu02);
}

#[test]
fn c1_rectangle_b5() {
    let r = &measure(&b1_and_b5())[1];
    assert_eq!(r.area, 20, "B5 (b): area of the 2 × 10 rectangle");
    assert!(close(r.mu20 / 20.0, 8.25, 1e-4) && close(r.mu02 / 20.0, 0.25, 1e-4) && close(r.mu11, 0.0, 1e-4), "B5 (b): S = diag(8.25, 0.25), got μ20/A = {}, μ11/A = {}, μ02/A = {}", r.mu20 / 20.0, r.mu11 / 20.0, r.mu02 / 20.0);
}

#[test]
fn c2_ellipse_b1() {
    let e = ellipse(&measure(&b1_and_b5())[0]);
    assert!(close(e.lambda1, 94.0 / 36.0, 1e-3), "B1 (d): λ1: got {}, expected 2.611", e.lambda1);
    assert!(close(e.lambda2, 12.0 / 36.0, 1e-3), "B1 (d): λ2: got {}, expected 0.333", e.lambda2);
    // v1 points right and down: 45° from x towards y (−45° counter-clockwise on screen)
    assert!(close(e.theta, PI / 4.0, 1e-3), "B1 (e): θ: got {}°, expected 45°", e.theta.to_degrees());
    assert!(close(e.eccentricity(), 0.934, 1e-3), "B1 (e): eccentricity: got {}, expected 0.934", e.eccentricity());
}

#[test]
fn c2_ellipse_b5() {
    let e = ellipse(&measure(&b1_and_b5())[1]);
    assert!(close(e.lambda1, 8.25, 1e-3) && close(e.lambda2, 0.25, 1e-3), "B5 (b): λ1, λ2: got {}, {}, expected 8.25, 0.25", e.lambda1, e.lambda2);
    assert!(close(e.theta, 0.0, 1e-3), "B5 (b): the rectangle lies along x: θ = {}°", e.theta.to_degrees());
    assert!(close(4.0 * e.lambda1.sqrt(), 11.49, 0.01), "B5 (b): major axis 4√λ1: got {}, expected 11.5", 4.0 * e.lambda1.sqrt());
}

#[test]
fn c2_rotation_keeps_the_eigenvalues() {
    // The B1 object turned by 90° (x ↔ y mirrored, then flipped): same λ, θ turned by 90°.
    let b1 = measure(&b1_and_b5())[0].clone();
    let turned = Image::from_fn(5, 5, |x, y| (b1_and_b5()[(y, 4 - x)] == 1) as u32);
    let (e, f) = (ellipse(&b1), ellipse(&measure(&turned)[0]));
    assert!(close(e.lambda1, f.lambda1, 1e-3) && close(e.lambda2, f.lambda2, 1e-3), "turning the object changed the eigenvalues: {e:?} and {f:?}");
    assert!(close(f.theta.abs(), PI / 4.0, 1e-3), "turned by 90°, the axis must be at ±45°: got {}°", f.theta.to_degrees());
}

#[test]
fn c3_edges_disc_b2() {
    let d = disc(4.0);
    let area = d.pixels().iter().filter(|&&v| v).count();
    assert_eq!(area, 49, "B2 (a): the digital disc of radius 4 has 49 pixels (the test image is wrong)");
    assert_eq!(perimeter_edges(&d), 36, "B2 (b): edge count of the disc of radius 4");
}

#[test]
fn c3_edges_square_b4() {
    let sq = Image::from_fn(12, 12, |x, y| (1..11).contains(&x) && (1..11).contains(&y));
    assert_eq!(perimeter_edges(&sq), 40, "B4 (a): an aligned 10 × 10 square has edge count 40");
    let full = Image::new(3, 2, true);
    assert_eq!(perimeter_edges(&full), 10, "a 3 × 2 image that is all object: the image border counts, 10");
}

#[test]
fn c4_crofton_disc() {
    // The edge count of a large disc is 4/π too long; Crofton is within about 1 %.
    let r = 30.0;
    let d = disc(r);
    let p = perimeter_crofton(&d);
    let edges = perimeter_edges(&d) as f32;
    let area = d.pixels().iter().filter(|&&v| v).count() as f32;
    let true_p = 2.0 * (PI * area).sqrt(); // the circle with the same area
    assert!(close(p / true_p, 1.0, 0.02), "Crofton on a disc of radius 30: got {p:.1}, expected about {true_p:.1}");
    assert!(close(edges / true_p, 4.0 / PI, 0.03), "edge count / true perimeter: got {:.3}, expected about 4/π = 1.273", edges / true_p);
}

#[test]
fn c4_crofton_rows_and_columns() {
    // n0 + n90 is exactly the edge count; the Crofton value is between 0.9 and 1.1 times the
    // edge count for a large aligned square (its sides are exact for the edge count).
    let sq = Image::from_fn(42, 42, |x, y| (1..41).contains(&x) && (1..41).contains(&y));
    let p = perimeter_crofton(&sq);
    assert!(close(p / 160.0, 1.0, 0.1), "Crofton on a 40 × 40 square: got {p:.1}, true 160");
}

#[test]
fn c5_bit_quads_b3() {
    let m = mask(&[
        ". . . . . . . . .",
        ". # # # . . . . .",
        ". # . # . . . . .",
        ". # # # . # # . .",
        ". . . . . . . # .",
        ". . . . . . . # .",
        ". . . . . . . . .",
    ]);
    assert_eq!(bit_quads(&m), (10, 4, 1), "B3 (b): (Q1, Q3, QD)");
    assert_eq!(euler_number(&m, true), 1, "B3: E8 = 2 objects − 1 hole");
    assert_eq!(euler_number(&m, false), 2, "B3: E4 = 3 objects − 1 hole");
}

#[test]
fn c5_euler_ring_and_discs() {
    let ring = Image::from_fn(9, 9, |x, y| {
        let r2 = (x as f32 - 4.0).powi(2) + (y as f32 - 4.0).powi(2);
        (4.0..=16.0).contains(&r2)
    });
    assert_eq!(euler_number(&ring, true), 0, "a ring: 1 object − 1 hole");
    let two = Image::from_fn(12, 5, |x, y| (1..4).contains(&y) && ((1..4).contains(&x) || (7..10).contains(&x)));
    assert_eq!(euler_number(&two, true), 2, "two blocks: 2 objects, no hole");
}
