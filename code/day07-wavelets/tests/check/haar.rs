// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day07 haar::

use day07::haar::*;

fn assert_close(got: &[f32], expected: &[f32], what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: got {} numbers, expected {}", got.len(), expected.len());
    for i in 0..got.len() {
        assert!((got[i] - expected[i]).abs() < 1e-3, "{what}: got {got:?}, expected {expected:?} (first difference at position {i})");
    }
}

const BOARD: [f32; 8] = [4.0, 6.0, 10.0, 12.0, 8.0, 6.0, 5.0, 5.0];

#[test]
fn c1_step_board_example() {
    let (s, d) = haar_step(&BOARD, false);
    assert_close(&s, &[5.0, 11.0, 7.0, 5.0], "level 1 averages of the board example");
    assert_close(&d, &[-1.0, -1.0, 1.0, 0.0], "level 1 differences of the board example");
}

#[test]
fn c1_full_transform_b1_b2() {
    let c = haar_forward(&[6.0, 2.0, 5.0, 7.0, 9.0, 9.0, 4.0, 0.0], false);
    assert_close(&c, &[5.25, -0.25, -1.0, 3.5, 2.0, -1.0, 0.0, 2.0], "B1, by hand");
    let c = haar_forward(&BOARD, false);
    assert_close(&c, &[7.0, 1.0, -3.0, 1.0, -1.0, -1.0, 1.0, 0.0], "board example, by hand");
    let r = std::f32::consts::SQRT_2;
    let c = haar_forward(&BOARD, true);
    assert_close(&c, &[19.799, 2.828, -6.0, 2.0, -r, -r, r, 0.0], "board example, orthonormal (B2)");
}

#[test]
fn c1_sparsity_b4_b5_b6() {
    let spike = haar_forward(&[0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0], false);
    assert_close(&spike, &[1.0, -1.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0], "spike at position 5 (B4)");
    let x = haar_forward(&[2.0, 2.0, 2.0, 10.0, 10.0, 10.0, 10.0, 10.0], false);
    assert_close(&x, &[7.0, -3.0, -2.0, 0.0, 0.0, -4.0, 0.0, 0.0], "step x (B5)");
    let y = haar_forward(&[2.0, 2.0, 2.0, 2.0, 10.0, 10.0, 10.0, 10.0], false);
    assert_close(&y, &[6.0, -4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "step y (B5)");
    let ramp = haar_forward(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], false);
    assert_close(&ramp, &[4.5, -2.0, -1.0, -1.0, -0.5, -0.5, -0.5, -0.5], "ramp (B6)");
}

#[test]
fn c2_inverse_undoes() {
    let c = [5.25, -0.25, -1.0, 3.5, 2.0, -1.0, 0.0, 2.0];
    assert_close(&haar_inverse(&c, false), &[6.0, 2.0, 5.0, 7.0, 9.0, 9.0, 4.0, 0.0], "inverse of B1");
    let x: Vec<f32> = (0..32).map(|i| ((i * 7) % 11) as f32 - 3.0).collect();
    for orthonormal in [false, true] {
        let back = haar_inverse(&haar_forward(&x, orthonormal), orthonormal);
        assert_close(&back, &x, &format!("forward then inverse (orthonormal: {orthonormal})"));
    }
}

#[test]
fn c2_energy_board_b2() {
    let c = haar_forward(&BOARD, true);
    let ex: f32 = BOARD.iter().map(|v| v * v).sum();
    let ec: f32 = c.iter().map(|v| v * v).sum();
    assert!((ex - 446.0).abs() < 1e-3 && (ec - 446.0).abs() < 1e-2, "orthonormal: Σx² = {ex}, Σc² = {ec}, both should be 446 (B2a)");
    let by_hand: f32 = haar_forward(&BOARD, false).iter().map(|v| v * v).sum();
    assert!((by_hand - 63.0).abs() < 1e-3, "by hand: Σc² = {by_hand}, expected 63 (energy is not kept)");
}

#[test]
fn c3_keep_three_b2() {
    let c = haar_forward(&BOARD, true);
    let (kept, dropped) = keep_largest(&c, 3);
    assert!((dropped - 10.0).abs() < 1e-3, "keep 3 of the board example: dropped energy {dropped}, expected 10 (B2b)");
    let xhat = haar_inverse(&kept, true);
    assert_close(&xhat, &[5.0, 5.0, 11.0, 11.0, 6.0, 6.0, 6.0, 6.0], "reconstruction from 3 coefficients (B2c)");
    let err: f32 = xhat.iter().zip(BOARD).map(|(a, b)| (a - b) * (a - b)).sum();
    assert!((err - dropped).abs() < 1e-3, "squared error {err} should equal the dropped energy {dropped}");
}

#[test]
fn c4_basis_is_orthonormal_b3() {
    let h = haar_basis(4);
    let r = std::f32::consts::FRAC_1_SQRT_2;
    let expected = [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, -0.5, -0.5], [r, -r, 0.0, 0.0], [0.0, 0.0, r, -r]];
    for i in 0..4 {
        assert_close(&h[i], &expected[i], &format!("row {i} of the 4 × 4 Haar matrix (B3)"));
    }
    let h = haar_basis(16);
    for i in 0..16 {
        for j in 0..16 {
            let dot: f32 = h[i].iter().zip(&h[j]).map(|(a, b)| a * b).sum();
            let e = if i == j { 1.0 } else { 0.0 };
            assert!((dot - e).abs() < 1e-5, "rows {i} and {j} of the 16 × 16 Haar matrix: dot product {dot}, expected {e}");
        }
    }
}
