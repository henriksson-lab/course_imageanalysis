// Checks for the coding exercises, using the numbers of the lecture and the pen-and-paper
// exercises. Run: cargo test -p svm maxmargin::

use svm::maxmargin::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// The lecture's example: blue (0, 0), (0, 2); red (2, 1), (3, 0). Street 0 ≤ x1 ≤ 2.
const LECTURE: [Point; 4] = [[0.0, 0.0], [0.0, 2.0], [2.0, 1.0], [3.0, 0.0]];
const LECTURE_LABELS: [f32; 4] = [-1.0, -1.0, 1.0, 1.0];

/// B1: blue (1, 1), (1, 3); red (4, 2), (5, 4). Street 1 ≤ x1 ≤ 4.
const B1: [Point; 4] = [[1.0, 1.0], [1.0, 3.0], [4.0, 2.0], [5.0, 4.0]];
const B1_LABELS: [f32; 4] = [-1.0, -1.0, 1.0, 1.0];

fn totals(ms: &[f32]) -> [f32; 4] {
    let mut t = [0.0; 4];
    for &m in ms {
        let l = losses(m);
        for k in 0..4 {
            t[k] += l[k];
        }
    }
    t
}

#[test]
fn c1_losses_lecture_table() {
    let t = totals(&[2.0, 1.0, 0.5, 0.0, -1.0]);
    let expected = [2.0, 3.5, 4.21, 6.25];
    for k in 0..4 {
        assert!(close(t[k], expected[k], 0.01), "totals [0–1, hinge, logistic, squared]: got {t:?}, expected {expected:?}");
    }
    let l = losses(-1.0);
    assert!(close(l[2], 1.90, 0.01), "logistic loss at y f = −1: got {}, expected 1.90 (in bits)", l[2]);
}

#[test]
fn c1_losses_b3() {
    let t = totals(&[3.0, 1.5, 1.0, 0.2, -0.5, -3.0]);
    let expected = [2.0, 6.3, 0.0, 23.14];
    for k in [0, 1, 3] {
        assert!(close(t[k], expected[k], 0.01), "B3 totals [0–1, hinge, _, squared]: got {t:?}, expected 2, 6.3, 23.14");
    }
}

#[test]
fn c1_slacks_b4() {
    // The slack of a point is its hinge loss.
    let ms = [2.3, 1.0, 0.7, -0.2, 1.5];
    let expected = [0.0, 0.0, 0.3, 1.2, 0.0];
    for (m, e) in ms.iter().zip(expected) {
        let h = losses(*m)[1];
        assert!(close(h, e, 1e-5), "B4: slack of the point with y f = {m}: got {h}, expected {e}");
    }
}

#[test]
fn c2_margins_b1() {
    // B1(c): w = (2/3, 0), b = −5/3 gives y f = 1, 1, 1, 5/3.
    let m = margins(&B1, &B1_LABELS, [2.0 / 3.0, 0.0], -5.0 / 3.0);
    let expected = [1.0, 1.0, 1.0, 5.0 / 3.0];
    for k in 0..4 {
        assert!(close(m[k], expected[k], 1e-4), "B1(c): got y f = {m:?}, expected {expected:?}");
    }
}

#[test]
fn c3_objective() {
    // The lecture's street: w = (1, 0), b = −1, no violations: ½‖w‖² = 0.5, for every C.
    for c in [0.1, 1.0, 1000.0] {
        let j = objective(&LECTURE, &LECTURE_LABELS, [1.0, 0.0], -1.0, c);
        assert!(close(j, 0.5, 1e-5), "lecture example, C = {c}: got {j}, expected 0.5");
    }
    // w = (0.5, 0), b = −0.5: a street twice as wide, but three points inside it (y f = 0.5, 0.5, 0.5).
    let j = objective(&LECTURE, &LECTURE_LABELS, [0.5, 0.0], -0.5, 1.0);
    assert!(close(j, 0.125 + 1.5, 1e-5), "w = (0.5, 0), b = −0.5, C = 1: got {j}, expected 0.125 + 3 · 0.5 = 1.625");
}

fn width(w: Point) -> f32 {
    2.0 / (w[0] * w[0] + w[1] * w[1]).sqrt()
}

#[test]
fn c4_train_lecture_example() {
    let (w, b) = train(&LECTURE, &LECTURE_LABELS, 10.0, 20000);
    assert!(close(w[0], 1.0, 0.03) && close(w[1], 0.0, 0.03) && close(b, -1.0, 0.03), "lecture example, C = 10: got w = {w:?}, b = {b}, expected w = (1, 0), b = −1");
    assert!(close(width(w), 2.0, 0.05), "lecture example: street width {}, expected 2", width(w));
}

#[test]
fn c4_train_b1() {
    let (w, b) = train(&B1, &B1_LABELS, 10.0, 20000);
    assert!(close(width(w), 3.0, 0.08), "B1: street width {}, expected 3 (w = {w:?}, b = {b})", width(w));
    let x = -b / w[0]; // where the line crosses x2 = 0 (it is vertical)
    assert!(close(x, 2.5, 0.08), "B1: the line should be x1 = 2.5, got w = {w:?}, b = {b}");
}

#[test]
fn c4_small_c_widens_the_street() {
    // S2: below C ≈ 0.5 the street of the lecture's example widens; at C = 0.1 all 4 points are inside it.
    let (w, b) = train(&LECTURE, &LECTURE_LABELS, 0.1, 20000);
    let m = margins(&LECTURE, &LECTURE_LABELS, w, b);
    assert!(width(w) > 3.0, "C = 0.1: the street should be much wider than 2, got {}", width(w));
    assert!(m.iter().all(|&m| m < 1.0), "C = 0.1: all four points should be inside the street, got y f = {m:?}");
}
