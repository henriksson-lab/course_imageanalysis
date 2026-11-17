// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p snakes

use imgcourse::Image;
use snakes::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// The square of B1: beads at (1, 1), (−1, 1), (−1, −1), (1, −1), in this order.
const SQUARE: [Point; 4] = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]];

#[test]
fn c1_elastic_square_b1() {
    let expected = [[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]];
    for (i, e) in expected.iter().enumerate() {
        let f = internal_force(&SQUARE, i, 1.0, 0.0);
        assert!(close(f[0], e[0], 1e-5) && close(f[1], e[1], 1e-5), "bead {i}: got {f:?}, expected {e:?}");
    }
}

#[test]
fn c1_rigid_square_b2() {
    let f = internal_force(&SQUARE, 0, 0.0, 1.0);
    assert!(close(f[0], -4.0, 1e-5) && close(f[1], -4.0, 1e-5), "rigid force at (1, 1): got {f:?}, expected [-4, -4]");
    let f = internal_force(&SQUARE, 0, 1.0, 1.0);
    assert!(close(f[0], -6.0, 1e-5) && close(f[1], -6.0, 1e-5), "α = β = 1 at (1, 1): got {f:?}, expected [-6, -6]");
}

#[test]
fn c1_octagon_b3() {
    let octagon = circle(0.0, 0.0, 1.0, 8);
    let f = internal_force(&octagon, 0, 1.0, 0.0);
    assert!(close(f[0], -0.586, 1e-3) && close(f[1], 0.0, 1e-5), "bead (1, 0): got {f:?}, expected [-0.586, 0]");
}

#[test]
fn c2_edge_map_of_a_ramp() {
    // A ramp has the same slope everywhere: the edge map is 1 everywhere away from the
    // border (after scaling to largest value 1), and blurring does not change a ramp.
    let ramp = Image::from_fn(40, 30, |x, _| 0.02 * x as f32);
    let f = edge_map(&ramp, 2.0);
    for (x, y) in [(20, 15), (10, 5), (30, 25)] {
        assert!(close(f[(x, y)], 1.0, 1e-3), "pixel ({x}, {y}): got {}, expected 1", f[(x, y)]);
    }
}

#[test]
fn c2_edge_map_of_a_step() {
    let step = imgcourse::samples::step(40, 20, 0.0, 1.0); // the step lies between x = 19 and 20
    let f = edge_map(&step, 1.5);
    assert!(f[(19, 10)] > 0.9 && f[(20, 10)] > 0.9, "at the step: got {} and {}, expected about 1", f[(19, 10)], f[(20, 10)]);
    assert!(f[(5, 10)] < 1e-3, "far from the step: got {}, expected about 0", f[(5, 10)]);
    assert!(f[(15, 10)] < f[(17, 10)] && f[(17, 10)] < f[(19, 10)], "f should rise towards the step");
}

#[test]
fn c3_one_step_without_image_b1_b2() {
    let flat = Image::new(5, 5, 0.0);
    let new = snake_step(&SQUARE, &flat, 1.0, 0.0, 1.0, 0.1);
    for (p, q) in SQUARE.iter().zip(&new) {
        assert!(close(q[0], 0.8 * p[0], 1e-5) && close(q[1], 0.8 * p[1], 1e-5), "B1(b): bead {p:?} went to {q:?}, expected 0.8 times it");
    }
    let new = snake_step(&SQUARE, &flat, 1.0, 1.0, 1.0, 0.1);
    assert!(close(new[0][0], 0.4, 1e-5) && close(new[0][1], 0.4, 1e-5), "B2(b): got {:?}, expected [0.4, 0.4]", new[0]);
}

#[test]
fn c3_image_force_b4() {
    // The edge map of B4 along every row: f(x) = exp(−(x − 10)²/8), the edge at x = 10.
    let f = Image::from_fn(30, 5, |x, _| (-((x as f32 - 10.0).powi(2)) / 8.0).exp());
    // Beads far apart, so that α = β = 0 leaves only the image force (w = 1, τ = 1).
    let beads = [[8.0, 2.0], [12.0, 2.0], [20.0, 2.0], [10.0, 2.0]];
    let new = snake_step(&beads, &f, 0.0, 0.0, 1.0, 1.0);
    let moved: Vec<f32> = beads.iter().zip(&new).map(|(p, q)| q[0] - p[0]).collect();
    assert!(close(moved[0], 0.3, 0.05), "bead at x = 8: moved {}, expected about +0.3 (towards the edge)", moved[0]);
    assert!(close(moved[1], -0.3, 0.05), "bead at x = 12: moved {}, expected about −0.3 (towards the edge)", moved[1]);
    assert!(moved[2].abs() < 1e-3, "bead at x = 20: moved {}, expected about 0 (outside the capture range)", moved[2]);
    assert!(moved[3].abs() < 1e-5, "bead on the edge: moved {}, expected 0", moved[3]);
    assert!(new.iter().all(|q| close(q[1], 2.0, 1e-5)), "no force along y expected: {new:?}");
}
