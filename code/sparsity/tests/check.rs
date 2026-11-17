// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p sparsity

use sparsity::*;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

/// The six coefficients of A1.
const W: [f64; 6] = [4.0, -0.5, 1.2, -2.5, 0.3, 3.0];

fn check_all(name: &str, f: fn(f64, f64) -> f64, expected: [f64; 6]) {
    let got: Vec<f64> = W.iter().map(|&w| f(w, 1.0)).collect();
    for (g, e) in got.iter().zip(expected) {
        assert!(close(*g, e), "{name} at λ = 1: got {got:?}, expected {expected:?}");
    }
}

#[test]
fn c1_three_penalties_a1() {
    check_all("ℓ2 (shrink_l2)", shrink_l2, [2.0, -0.25, 0.6, -1.25, 0.15, 1.5]);
    check_all("ℓ1 (soft)", soft, [3.0, 0.0, 0.2, -1.5, 0.0, 2.0]);
    check_all("ℓ0 (hard)", hard, [4.0, 0.0, 1.2, -2.5, 0.0, 3.0]);
}

#[test]
fn c1_energy_of_the_l2_answer_a1e() {
    let e: f64 = W.iter().map(|&w| shrink_l2(w, 1.0).powi(2)).sum();
    assert!((e - 8.2575).abs() < 1e-6, "A1(e): energy of the ℓ2 answer: got {e}, expected 33.03/4 = 8.26");
}

#[test]
fn c1_thresholds_at_exactly_lambda() {
    // At |w| = λ exactly both thresholds give 0.
    assert!(close(soft(1.0, 1.0), 0.0) && close(soft(-1.0, 1.0), 0.0), "soft(±1, 1) should be 0");
    assert!(close(hard(1.0, 1.0), 0.0), "hard(1, 1) should be 0, got {}", hard(1.0, 1.0));
}

#[test]
fn c2_count_for_energy_a2() {
    let c = [10.0, 6.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.2, 0.0, 0.0];
    assert_eq!(count_for_energy(&c, 0.95), 3, "A2(b): 95 % of the energy needs 3 coefficients");
    assert_eq!(count_for_energy(&c, 0.99), 4, "A2(b): 99 % of the energy needs 4 coefficients");
    // The order and the signs must not matter.
    let shuffled = [0.0, -2.0, 0.3, 10.0, -0.5, 1.0, 0.0, 6.0, -3.0, 0.2];
    assert_eq!(count_for_energy(&shuffled, 0.95), 3, "sort the magnitudes first: same signal, shuffled and with signs");
}

#[test]
fn c3_lambda_max_a4_and_e2() {
    let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let l = lambda_max(&identity, &[3.0, 1.0]);
    assert!(close(l, 3.0), "A4(a): A = I, y = (3, 1): got {l}, expected 3");
    // Explore E2: columns with correlation 0.3 (AᵀA = [[1, 0.3], [0.3, 1]]) and least-squares
    // point (3, 1), so y = A (3, 1).
    let s = (1.0f64 - 0.09).sqrt();
    let a = vec![vec![1.0, 0.3], vec![0.0, s]];
    let y = [3.0 + 0.3, s];
    let l = lambda_max(&a, &y);
    assert!((l - 3.3).abs() < 1e-9, "E2: ρ = 0.3: got {l}, expected 3.3");
}

#[test]
fn c4_project_l1_ball_a3() {
    let p = project_l1_ball(&[3.0, 1.0], 2.0);
    assert!(close(p[0], 2.0) && close(p[1], 0.0), "A3(b): budget 2: got {p:?}, expected the corner (2, 0)");
    let p = project_l1_ball(&[3.0, 1.0], 3.0);
    assert!(close(p[0], 2.5) && close(p[1], 0.5), "A3(c): budget 3: got {p:?}, expected (2.5, 0.5), two nonzeros");
    let p = project_l1_ball(&[3.0, 1.0], 5.0);
    assert!(close(p[0], 3.0) && close(p[1], 1.0), "budget 5: (3, 1) is inside the ball, got {p:?}");
    let p = project_l1_ball(&[-1.0, 4.0, 0.5, -2.0], 3.0);
    let norm: f64 = p.iter().map(|x| x.abs()).sum();
    assert!(close(norm, 3.0), "the result must lie on the ball: ℓ1 norm got {norm}, expected 3 ({p:?})");
    assert!(close(p[0], 0.0) && close(p[1], 2.5) && close(p[2], 0.0) && close(p[3], -0.5), "got {p:?}, expected (0, 2.5, 0, −0.5)");
}
