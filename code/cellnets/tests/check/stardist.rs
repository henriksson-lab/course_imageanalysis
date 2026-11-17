// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p cellnets stardist::

use imgcourse::Image;
use cellnets::stardist::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// B1: an 8 × 4 rectangle (label 1) inside a 12 × 8 image (label 0 around it). Pixel (x, y)
/// covers x − 0.5 ... x + 0.5, so the rectangle covers 1.5 ... 9.5 by 1.5 ... 5.5 and its
/// centre is (5.5, 3.5).
fn rectangle() -> Image<u32> {
    Image::from_fn(12, 8, |x, y| if (2..10).contains(&x) && (2..6).contains(&y) { 1 } else { 0 })
}

#[test]
fn c1_rays_of_the_rectangle_b1() {
    let r = rays(&rectangle(), 5.5, 3.5, 8);
    let d = 2.0 * 2f32.sqrt();
    let expected = [4.0, d, 2.0, d, 4.0, d, 2.0, d];
    for k in 0..8 {
        assert!(close(r[k], expected[k], 0.02), "ray {k}: got {}, expected {:.2} (B1a)", r[k], expected[k]);
    }
}

#[test]
fn c1_rays_stop_at_another_label() {
    // two touching objects: the rays of object 1 stop where object 2 begins
    let labels = Image::from_fn(10, 3, |x, _| if x < 5 { 1 } else { 2 });
    let r = rays(&labels, 2.0, 1.0, 4);
    assert!(close(r[0], 2.5, 0.02), "ray to the right: got {}, expected 2.5 (object 2 starts at x = 4.5)", r[0]);
    assert!(close(r[1], 1.5, 0.02), "ray down: got {}, expected 1.5 (the image ends at y = 2.5)", r[1]);
}

#[test]
fn c2_area_b1_b2() {
    let d = 2.0 * 2f32.sqrt();
    let a = polygon_area(&[4.0, d, 2.0, d, 4.0, d, 2.0, d]);
    assert!(close(a, 24.0, 0.01), "B1b: got {a}, expected 24.0");
    let a = polygon_area(&[5.0, 4.0, 3.0, 4.0, 5.0, 4.0, 3.0, 4.0]);
    assert!(close(a, 45.25, 0.01), "B2b: got {a}, expected 45.3");
}

#[test]
fn c2_area_of_a_disc_b6() {
    // a disc of radius 1 from its centre: a fraction n/(2π) sin(2π/n) of π
    let a8 = polygon_area(&[1.0; 8]) / std::f32::consts::PI;
    let a32 = polygon_area(&[1.0; 32]) / std::f32::consts::PI;
    assert!(close(a8, 0.900, 0.001), "8 rays: got {a8}, expected 0.900 of the disc (B6)");
    assert!(close(a32, 0.994, 0.001), "32 rays: got {a32}, expected 0.994 of the disc (B6)");
}

#[test]
fn c3_inside_a_square_and_a_triangle() {
    let square = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)];
    assert!(inside_polygon(2.0, 2.0, &square), "(2, 2) should be inside the square 0..4");
    assert!(!inside_polygon(5.0, 2.0, &square), "(5, 2) should be outside the square 0..4");
    assert!(!inside_polygon(-1.0, 2.0, &square), "(-1, 2) should be outside the square 0..4");
    let triangle = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)];
    assert!(inside_polygon(2.0, 3.0, &triangle), "(2, 3) should be inside the triangle");
    assert!(!inside_polygon(6.0, 6.0, &triangle), "(6, 6) should be outside the triangle");
}

#[test]
fn c4_iou_rectangle_8_rays_e7() {
    // E7a: the rectangle of B1 scaled by 5 (40 × 20 pixels); its 8-ray polygon covers 600 of
    // 800 pixels, IoU 0.75. Centre at (19.5, 9.5), so that the rectangle is pixels 0..40 × 0..20.
    let (cx, cy) = (19.5, 9.5);
    let rect = [(-0.5, -0.5), (39.5, -0.5), (39.5, 19.5), (-0.5, 19.5)];
    let d = 10.0 * 2f32.sqrt();
    let poly = polygon_vertices(cx, cy, &[20.0, d, 10.0, d, 20.0, d, 10.0, d]);
    let iou = polygon_iou(&rect, &poly);
    assert!(close(iou, 0.75, 0.02), "8-ray polygon against the 40 × 20 rectangle: got IoU {iou}, expected 0.75 (E7a)");
    let same = polygon_iou(&poly, &poly);
    assert!(close(same, 1.0, 1e-6), "a polygon with itself: got IoU {same}, expected 1");
    let far: Vec<(f32, f32)> = poly.iter().map(|&(x, y)| (x + 100.0, y)).collect();
    let none = polygon_iou(&poly, &far);
    assert!(close(none, 0.0, 1e-6), "polygons far apart: got IoU {none}, expected 0");
}

/// The six candidates of B4 and their IoUs (P1 ... P6 are indices 0 ... 5).
fn b4() -> ([f32; 6], impl Fn(usize, usize) -> f32) {
    let scores = [0.92, 0.85, 0.80, 0.62, 0.55, 0.35];
    let pairs = [((0, 1), 0.71), ((0, 2), 0.18), ((0, 3), 0.05), ((0, 4), 0.33), ((1, 2), 0.22), ((1, 4), 0.41), ((2, 3), 0.46), ((2, 5), 0.12), ((3, 5), 0.58)];
    let iou = move |i: usize, j: usize| {
        let (a, b) = (i.min(j), i.max(j));
        pairs.iter().find(|p| p.0 == (a, b)).map_or(0.0, |p| p.1)
    };
    (scores, iou)
}

#[test]
fn c5_nms_b4() {
    let (scores, iou) = b4();
    let cases = [(0.5, 0.3, vec![0, 2]), (0.5, 0.5, vec![0, 2, 3, 4]), (0.3, 0.7, vec![0, 2, 3, 4, 5])];
    for (prob, t, expected) in cases {
        let kept = nms(&scores, prob, t, &iou);
        let names: Vec<String> = kept.iter().map(|i| format!("P{}", i + 1)).collect();
        let want: Vec<String> = expected.iter().map(|i| format!("P{}", i + 1)).collect();
        assert_eq!(kept, expected, "probability {prob}, NMS {t}: kept {names:?}, expected {want:?} (B4)");
    }
}
