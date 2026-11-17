// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p polar-boundary
//
// A polar cost table is an image with one column per angle and one row per radius. The
// exercise counts radii from 1, the code from 0: radius r is row r − 1.

use imgcourse::Image;
use polar_boundary::*;

/// The cost table of B1: angles 0°, 90°, 180°, 270° across, radii 1 to 4 down.
fn b1() -> Image<f32> {
    Image::from_rows(&[[3., 9., 1., 0.], [6., 0., 2., 6.], [1., 5., 9., 6.], [1., 3., 3., 7.]])
}

/// The cost table of B7.
fn b7() -> Image<f32> {
    Image::from_rows(&[[3., 8., 9., 0.], [2., 4., 1., 3.], [5., 4., 0., 5.], [2., 1., 1., 4.]])
}

fn radii_from_1(path: &[usize]) -> Vec<usize> {
    path.iter().map(|r| r + 1).collect()
}

#[test]
fn c1_bilinear_b2() {
    // B2 (b): the four pixels around (54.33, 42.5).
    let mut img = Image::new(60, 50, 0.0);
    img[(54, 42)] = 100.0;
    img[(55, 42)] = 120.0;
    img[(54, 43)] = 140.0;
    img[(55, 43)] = 180.0;
    let a = 30f32.to_radians();
    let v = bilinear(&img, 50.0 + 5.0 * a.cos(), 40.0 + 5.0 * a.sin());
    assert!((v - 129.90).abs() < 0.01, "B2 (b): got {v}, expected 129.90");
    assert!((bilinear(&img, 54.0, 42.0) - 100.0).abs() < 1e-4, "at a pixel centre: got {}, expected the pixel, 100", bilinear(&img, 54.0, 42.0));
}

#[test]
fn c2_unwrap_b2() {
    // B2 (a): from the centre (50, 40), the sample at r = 10, θ = 90° reads pixel (50, 50).
    let mut img = Image::new(100, 100, 0.0);
    img[(50, 50)] = 1.0;
    let polar = unwrap(&img, 50.0, 40.0, 4, 12);
    assert_eq!((polar.width(), polar.height()), (4, 12), "size: 4 angles across, 12 radii down");
    assert!((polar[(1, 10)] - 1.0).abs() < 1e-3, "angle 90° (column 1), r = 10: got {}, expected 1 (pixel (50, 50), straight down)", polar[(1, 10)]);
    assert!(polar[(3, 10)].abs() < 1e-3, "angle 270° (column 3), r = 10: got {}, expected 0 (that is pixel (50, 30))", polar[(3, 10)]);
}

#[test]
fn c3_polarity_b3() {
    // B3: one ray, nucleus 160, then the brighter neighbour 220, then background 40.
    let ray = [160., 160., 160., 220., 220., 220., 40., 40., 40.];
    let polar = Image::from_vec(1, ray.len(), ray.to_vec());
    let argmin = |c: &Image<f32>| (0..c.height()).min_by(|&a, &b| c[(0, a)].total_cmp(&c[(0, b)])).unwrap();
    let b2d = radial_cost(&polar, Polarity::BrightToDark);
    let r = argmin(&b2d);
    assert!(r == 5 || r == 6, "bright → dark: cheapest at radius index {r}, expected the neighbour's outer edge (5 or 6)");
    assert!(b2d[(0, 3)] > 0.0, "bright → dark at the contact 160 → 220: got {}, expected a positive (expensive) cost", b2d[(0, 3)]);
    let d2b = radial_cost(&polar, Polarity::DarkToBright);
    let r = argmin(&d2b);
    assert!(r == 2 || r == 3, "dark → bright: cheapest at radius index {r}, expected the contact (2 or 3)");
    let either = radial_cost(&polar, Polarity::Either);
    assert_eq!(argmin(&either), argmin(&b2d), "either: the larger jump (220 → 40) should be cheapest");
    assert!((b2d[(0, 5)] + 90.0).abs() < 1e-4, "bright → dark at radius index 5: got {}, expected (40 − 220)/2 = −90", b2d[(0, 5)]);
}

#[test]
fn c4_table_b1() {
    let (m, _) = polar_table(&b1(), 1, 0.0, None);
    let expected = [[3., 12., 2., 2.], [6., 1., 3., 8.], [1., 6., 10., 9.], [1., 4., 7., 14.]];
    for (r, row) in expected.iter().enumerate() {
        for (k, &e) in row.iter().enumerate() {
            assert_eq!(m[(k, r)], e, "M at angle {}°, radius {}: got {}, expected {e}", 90 * k, r + 1, m[(k, r)]);
        }
    }
    // B1 (c): with the start fixed at radius 1, the last column is (4, 10, 11, 19).
    let (m, _) = polar_table(&b1(), 1, 0.0, Some(0));
    let got: Vec<f32> = (0..4).map(|r| m[(3, r)]).collect();
    assert_eq!(got, vec![4., 10., 11., 19.], "start radius 1, M at 270°: got {got:?}");
}

#[test]
fn c5_open_b1() {
    let (path, cost) = open_path(&b1(), 1, 0.0);
    assert_eq!(radii_from_1(&path), vec![3, 2, 1, 1], "B1 (a): the open path (radii 1–4): got {:?}", radii_from_1(&path));
    assert_eq!(cost, 2.0, "B1 (a): its cost: got {cost}, expected 2");
}

#[test]
fn c5_closed_b1() {
    let (path, cost) = closed_path(&b1(), 1, 0.0);
    assert_eq!(radii_from_1(&path), vec![1, 2, 1, 1], "B1 (c): the closed path: got {:?}", radii_from_1(&path));
    assert_eq!(cost, 4.0, "B1 (c): its cost: got {cost}, expected 4");
}

#[test]
fn c5_unroll_counterexample_b7() {
    let (path, cost) = open_path(&b7(), 1, 0.0);
    assert_eq!(radii_from_1(&path), vec![4, 4, 3, 2], "B7: the open path: got {:?}", radii_from_1(&path));
    assert_eq!(cost, 6.0, "B7: open cost: got {cost}, expected 6");
    let (path, cost) = closed_path(&b7(), 1, 0.0);
    assert_eq!(cost, 7.0, "B7: the best closed path costs 7, got {cost} for {:?}", radii_from_1(&path));
    assert!(path[0].abs_diff(path[3]) <= 1, "B7: the closed path must close: got {:?}", radii_from_1(&path));
}

#[test]
fn c5_closure_with_penalty() {
    // With alpha > 0 the closing step is paid too: a path that jumps back costs more.
    let cost = Image::from_rows(&[[0., 0., 9., 9.], [9., 9., 0., 0.]]);
    let (path, total) = closed_path(&cost, 1, 0.5);
    // Path (0, 0, 1, 1): 0 + 0.5 (step) + 0 + 0.5 (closing step) = 1.
    assert_eq!(path, vec![0, 0, 1, 1], "got {path:?}");
    assert!((total - 1.0).abs() < 1e-6, "cost with the closing step: got {total}, expected 1");
}
