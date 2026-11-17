// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day14 shape_features::

use imgcourse::Image;
use day14::shape_features::*;

/// A mask written row by row: "1" = object, anything else = background.
fn mask(rows: &[&str]) -> Image<bool> {
    let cells: Vec<Vec<bool>> = rows.iter().map(|r| r.split_whitespace().map(|c| c == "1").collect()).collect();
    Image::from_fn(cells[0].len(), cells.len(), |x, y| cells[y][x])
}

fn b1() -> Image<bool> {
    mask(&["1 1 0 0 0 0", "1 1 0 0 1 1", "1 1 1 1 1 1", "1 1 1 1 1 1", "0 0 1 1 0 0", "0 0 1 1 0 0"])
}

const B2: [Point; 5] = [(0., 0.), (6., 0.), (8., 3.), (5., 6.), (0., 4.)];

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn c1_candidates_b1() {
    let c = hull_candidates(&b1());
    assert_eq!(c.len(), 24, "6 rows, 4 corners each: got {} points", c.len());
    for p in [(0., 0.), (2., 0.), (6., 1.), (2., 6.), (4., 6.)] {
        assert!(c.contains(&p), "B1 (a): the corner {p:?} must be a candidate; got {c:?}");
    }
}

#[test]
fn c2_hull_b1() {
    let hull = convex_hull(&hull_candidates(&b1()));
    let expected = vec![(0., 0.), (2., 0.), (6., 1.), (6., 4.), (4., 6.), (2., 6.), (0., 4.)];
    assert_eq!(hull, expected, "B1 (a): the hull of the pixel corners");
}

#[test]
fn c2_hull_drops_inside_and_middle_points() {
    let pts = [(0., 0.), (2., 0.), (4., 0.), (4., 4.), (2., 2.), (1., 3.), (0., 4.), (0., 2.)];
    let hull = convex_hull(&pts);
    assert_eq!(hull, vec![(0., 0.), (4., 0.), (4., 4.), (0., 4.)], "a square with points inside and on its edges");
}

#[test]
fn c3_area_b1_b2() {
    let hull = convex_hull(&hull_candidates(&b1()));
    let a = polygon_area(&hull);
    assert!(close(a, 30.0, 1e-4), "B1 (b): hull area: got {a}, expected 30");
    let solidity = 22.0 / a;
    assert!(close(solidity, 0.733, 1e-3), "B1 (b): solidity: got {solidity}, expected 0.733");
    let a2 = polygon_area(&B2);
    assert!(close(a2, 35.5, 1e-4), "B2 (a): area: got {a2}, expected 35.5");
    let reversed: Vec<Point> = B2.iter().rev().copied().collect();
    assert!(close(polygon_area(&reversed), 35.5, 1e-4), "the area must not depend on the order of the vertices");
}

#[test]
fn c3_centre_hull_b1() {
    // B1 (c): the hull of the pixel centres has area 19, "solidity" 22/19 > 1.
    let m = b1();
    let centres: Vec<Point> = m.coords().filter(|&p| m[p]).map(|(x, y)| (x as f32 + 0.5, y as f32 + 0.5)).collect();
    let a = polygon_area(&convex_hull(&centres));
    assert!(close(a, 19.0, 1e-4), "B1 (c): area of the hull of the centres: got {a}, expected 19");
}

#[test]
fn c4_feret_b2() {
    let (min, max) = feret(&B2);
    assert!(close(min, 32.0 / 29f32.sqrt(), 1e-3), "B2 (b): min Feret: got {min}, expected 5.94");
    assert!(close(max, 73f32.sqrt(), 1e-3), "B2 (c): max Feret: got {max}, expected 8.54");
}

#[test]
fn c4_feret_b1() {
    // Exercise E1: the corner hull of B1 has max Feret √52 = 7.21 and min Feret 5.82.
    let (min, max) = feret(&convex_hull(&hull_candidates(&b1())));
    assert!(close(max, 52f32.sqrt(), 1e-3), "B1 hull: max Feret: got {max}, expected 7.21");
    assert!(close(min, 5.82, 0.01), "B1 hull: min Feret: got {min}, expected 5.82");
}

#[test]
fn c5_distance_transform_b3() {
    let m = mask(&[
        "0 0 0 0 0 0 0 0 0",
        "0 1 1 1 1 1 1 1 0",
        "0 1 1 1 1 1 1 1 0",
        "0 1 1 1 1 1 1 1 0",
        "0 1 1 1 1 0 0 0 0",
        "0 1 1 1 1 0 0 0 0",
        "0 0 0 0 0 0 0 0 0",
    ]);
    let dt = distance_transform(&m);
    // B3: row 3, column 3 is (x, y) = (3, 3)
    assert!(close(dt[(3, 3)], 5f32.sqrt(), 1e-4), "B3: DT at row 3, column 3: got {}, expected √5 = 2.24", dt[(3, 3)]);
    assert!(close(dt[(4, 3)], 2f32.sqrt(), 1e-4), "B3 (c): DT right of it: got {}, expected √2 = 1.41", dt[(4, 3)]);
    assert!(close(dt.max(), 5f32.sqrt(), 1e-4), "B3 (a): largest DT: got {}, expected 2.24", dt.max());
    let n = m.pixels().iter().filter(|&&v| v).count();
    let mean = dt.sum() / n as f32;
    assert_eq!(n, 29, "B3 has 29 object pixels (the test image is wrong)");
    assert!(close(mean, 38.65 / 29.0, 0.01), "B3 (b): mean DT: got {mean}, expected 1.33");
}

#[test]
fn c5_outside_counts_as_background() {
    let dt = distance_transform(&Image::new(5, 3, true));
    assert!(close(dt[(2, 1)], 2.0, 1e-4) && close(dt[(0, 0)], 1.0, 1e-4), "an image that is all object: the centre is 2 from outside, a corner 1; got {} and {}", dt[(2, 1)], dt[(0, 0)]);
}
