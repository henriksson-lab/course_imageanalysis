// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p dynprog fastmarching::

use dynprog::fastmarching::*;
use imgcourse::Image;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_updates_b1() {
    let cases = [
        ((2.0, 2.5, 1.0), 2.911, "B1 (a)"),
        ((2.0, 2.5, 2.0), 3.642, "B1 (b), P = 2"),
        ((2.0, f32::INFINITY, 1.0), 3.0, "B1 (c), only the left neighbour known"),
        ((2.0, 3.2, 1.0), 3.0, "B1 (d), the upper neighbour too late"),
    ];
    for ((a, b, p), expected, name) in cases {
        let t = upwind_update(a, b, p);
        assert!(close(t, expected, 1e-3), "{name}: got {t}, expected {expected}");
    }
}

#[test]
fn c1_diagonal_front_b3() {
    let t = upwind_update(5.0, 5.0, 1.0);
    assert!(close(t, 5.0 + 0.5f32.sqrt(), 1e-5), "B3: a = b = 5: got {t}, expected 5.707");
}

#[test]
fn c2_quadrant_b2() {
    let t = fast_marching(&Image::new(4, 4, 1.0), &[(0, 0)]);
    let expected = [((1, 0), 1.0), ((1, 1), 1.707), ((2, 1), 2.545), ((2, 2), 3.252), ((3, 1), 3.442), ((3, 2), 4.048), ((3, 3), 4.755)];
    for ((x, y), e) in expected {
        assert!(close(t[(x, y)], e, 2e-3), "B2: T({x}, {y}): got {}, expected {e}", t[(x, y)]);
        assert!(close(t[(y, x)], t[(x, y)], 1e-5), "B2: T should be symmetric, T({x}, {y}) = {} but T({y}, {x}) = {}", t[(x, y)], t[(y, x)]);
    }
}

#[test]
fn c2_two_seeds_b5() {
    // B5: seeds at (70, 90) and (190, 170): at the midpoint (130, 130), T ≈ 72.11.
    let t = fast_marching(&Image::new(261, 261, 1.0), &[(70, 90), (190, 170)]);
    let v = t[(130, 130)];
    assert!(v > 72.11 - 0.5 && v < 72.11 + 2.0, "B5: T(130, 130): got {v}, expected about 72.11 (a little more: point seeds)");
}

#[test]
fn c3_dijkstra_b2() {
    let flat = Image::new(4, 4, 1.0);
    let d4 = dijkstra_time(&flat, &[(0, 0)], false);
    let d8 = dijkstra_time(&flat, &[(0, 0)], true);
    for ((x, y), e4, e8) in [((1, 1), 2.0, 1.414), ((2, 1), 3.0, 2.414), ((2, 2), 4.0, 2.828), ((3, 1), 4.0, 3.414), ((3, 3), 6.0, 4.243)] {
        assert!(close(d4[(x, y)], e4, 1e-3), "B2: Dijkstra 4 at ({x}, {y}): got {}, expected {e4}", d4[(x, y)]);
        assert!(close(d8[(x, y)], e8, 1e-3), "B2: Dijkstra 8 at ({x}, {y}): got {}, expected {e8}", d8[(x, y)]);
    }
}

#[test]
fn c3_cost_is_per_length() {
    // Twice the cost per pixel: twice the time.
    let d = dijkstra_time(&Image::new(5, 5, 2.0), &[(0, 0)], false);
    assert!(close(d[(4, 4)], 16.0, 1e-4), "cost 2, 8 steps: got {}, expected 16", d[(4, 4)]);
}

#[test]
fn c4_descent_is_straight_b9() {
    // B9: with P = 1 and one seed, the descent runs straight to the seed.
    let t = fast_marching(&Image::new(60, 50, 1.0), &[(5, 5)]);
    let path = descend(&t, (45.0, 35.0), 0.5);
    let &(xe, ye) = path.last().unwrap();
    assert!((xe - 5.0).abs() <= 0.5 && (ye - 5.0).abs() <= 0.5, "the path should end at the seed (5, 5), ends at ({xe:.2}, {ye:.2})");
    // distance of every point from the line through (5, 5) and (45, 35)
    let (dx, dy) = (40.0f32, 30.0f32);
    for &(x, y) in &path {
        let dist = ((x - 5.0) * dy - (y - 5.0) * dx).abs() / (dx * dx + dy * dy).sqrt();
        assert!(dist < 1.5, "the point ({x:.2}, {y:.2}) is {dist:.2} px off the straight line to the seed");
    }
    assert!(path.len() < 120, "a 50 px path in steps of 0.5 should take about 100 steps, took {}", path.len());
}
