// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p svm classify::

use svm::classify::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// B1: f(x) = x1 + 2 x2 − 4 and the points A, B, C, D.
const W: Point = [1.0, 2.0];
const B: f32 = -4.0;
const ABCD: [Point; 4] = [[1.0, 1.0], [3.0, 2.0], [0.0, 3.0], [2.0, 0.5]];

/// B3: three points of class +1, then three of class −1.
const B3: [Point; 6] = [[3.0, 3.0], [4.0, 2.0], [5.0, 4.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0]];
const B3_LABELS: [f32; 6] = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0];

#[test]
fn c1_decision_b1() {
    let expected = [-1.0, 3.0, 2.0, -1.0];
    for (name, (x, e)) in ["A", "B", "C", "D"].iter().zip(ABCD.iter().zip(expected)) {
        let f = decision(W, B, *x);
        assert!(close(f, e, 1e-5), "f({name}): got {f}, expected {e}");
    }
}

#[test]
fn c1_signed_distance_b1() {
    let expected = [-0.4472, 1.3416, 0.8944, -0.4472];
    for (name, (x, e)) in ["A", "B", "C", "D"].iter().zip(ABCD.iter().zip(expected)) {
        let d = signed_distance(W, B, *x);
        assert!(close(d, e, 1e-3), "distance of {name}: got {d}, expected {e}");
    }
}

#[test]
fn c1_scaling_b2() {
    // (3w, 3b): f triples, the distances stay. (−w, −b): the distances change sign.
    for x in ABCD {
        let (f, d) = (decision(W, B, x), signed_distance(W, B, x));
        let f3 = decision([3.0, 6.0], -12.0, x);
        let d3 = signed_distance([3.0, 6.0], -12.0, x);
        let dm = signed_distance([-1.0, -2.0], 4.0, x);
        assert!(close(f3, 3.0 * f, 1e-4), "f with (3w, 3b) at {x:?}: got {f3}, expected {}", 3.0 * f);
        assert!(close(d3, d, 1e-4), "distance with (3w, 3b) at {x:?}: got {d3}, expected {d}");
        assert!(close(dm, -d, 1e-4), "distance with (−w, −b) at {x:?}: got {dm}, expected {}", -d);
    }
}

#[test]
fn c2_smallest_margin() {
    // B1 with labels from the signs: A and D (−1) are closest, at 0.447.
    let labels = [-1.0, 1.0, 1.0, -1.0];
    let m = smallest_margin(&ABCD, &labels, W, B);
    assert!(close(m, 0.4472, 1e-3), "B1, all correct: got {m}, expected 0.447");
    // Call B a −1 point: it is 1.342 on the wrong side.
    let m = smallest_margin(&ABCD, &[-1.0, -1.0, 1.0, -1.0], W, B);
    assert!(close(m, -1.3416, 1e-3), "B1 with B labelled −1: got {m}, expected −1.342");
}

#[test]
fn c3_nearest_mean_b3() {
    let (w, b) = nearest_mean(&B3, &B3_LABELS);
    assert!(close(w[0], 3.0, 1e-4) && close(w[1], 2.0, 1e-4), "w: got {w:?}, expected [3, 2]");
    assert!(close(b, -11.5, 1e-4), "b: got {b}, expected −11.5");
    assert_eq!(count_errors(&B3, &B3_LABELS, w, b), 0, "B3(b): all six training points should be correct");
    let f1 = decision(w, b, [2.0, 3.0]);
    let f2 = decision(w, b, [3.0, 1.0]);
    assert!(close(f1, 0.5, 1e-4), "B3(c): f(2, 3) = {f1}, expected 0.5");
    assert!(close(f2, -0.5, 1e-4), "B3(c): f(3, 1) = {f2}, expected −0.5");
}

#[test]
fn c4_perceptron_separates_b3() {
    let (w, b, updates) = perceptron(&B3, &B3_LABELS, 1000);
    assert!(updates > 0, "the perceptron should need at least one update");
    let errors = count_errors(&B3, &B3_LABELS, w, b);
    assert_eq!(errors, 0, "B3 is separable: the perceptron should stop at a separating line, got w = {w:?}, b = {b} with {errors} errors");
}

#[test]
fn c4_perceptron_one_update_b9() {
    // One point, wrong at the start (f = 0): one update raises y f by ‖x‖² + 1 (B9).
    let x = [2.0, 1.0];
    let (w, b, updates) = perceptron(&[x], &[1.0], 1);
    let yf = decision(w, b, x);
    assert_eq!(updates, 1, "one point, one pass: expected 1 update, got {updates}");
    assert!(close(yf, 6.0, 1e-5), "y f after one update: got {yf}, expected ‖x‖² + 1 = 6");
}

#[test]
fn c4_perceptron_xor_b5() {
    // XOR (B5): no line works, so the perceptron never stops; it must give up after max_passes.
    let xor = [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]];
    let labels = [-1.0, -1.0, 1.0, 1.0];
    let (w, b, updates) = perceptron(&xor, &labels, 50);
    assert!(updates >= 50, "XOR: expected an update in every one of the 50 passes, got {updates}");
    assert!(count_errors(&xor, &labels, w, b) >= 1, "XOR: no line can classify all four points");
}
