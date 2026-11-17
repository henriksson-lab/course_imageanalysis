// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day07 filterbank::

use day07::filterbank::*;

fn assert_close(got: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: got {} numbers, expected {}", got.len(), expected.len());
    for i in 0..got.len() {
        assert!((got[i] - expected[i]).abs() < tol, "{what}: got {got:?}, expected {expected:?} (first difference at position {i})");
    }
}

#[test]
fn c1_highpass() {
    let r = std::f32::consts::FRAC_1_SQRT_2;
    assert_close(&highpass(&HAAR), &[r, -r], 1e-6, "Haar high-pass");
    assert_close(&highpass(&DB2), &[-0.1294, -0.2241, 0.8365, -0.4830], 1e-4, "db2 high-pass (table of the exercise)");
}

#[test]
fn c2_vanishing_moments_b2() {
    assert_eq!(vanishing_moments(&highpass(&HAAR)), 1, "Haar has 1 vanishing moment");
    assert_eq!(vanishing_moments(&highpass(&DB2)), 2, "db2 has 2 vanishing moments (B2)");
    assert_eq!(vanishing_moments(&highpass(&DB3)), 3, "db3 has 3 vanishing moments");
    assert_eq!(vanishing_moments(&highpass(&DB4)), 4, "db4 has 4 vanishing moments");
}

#[test]
fn c3_analysis_b1_b3() {
    let (a, d) = analysis(&[4.0, 6.0, 10.0, 12.0], &HAAR);
    assert_close(&a, &[7.071, 15.556], 1e-3, "Haar approximations of (4, 6, 10, 12) (B1a)");
    assert_close(&d, &[-1.414, -1.414], 1e-3, "Haar details of (4, 6, 10, 12) (B1a)");
    let ramp: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let (_, d) = analysis(&ramp, &DB2);
    assert_close(&d, &[0.0, 0.0, 0.0, -2.828], 1e-3, "db2 details of the ramp 0..7, periodic (B3)");
    let (_, d) = analysis(&ramp, &HAAR);
    assert_close(&d, &[-std::f32::consts::FRAC_1_SQRT_2; 4], 1e-4, "Haar details of the ramp (B3c)");
}

#[test]
fn c4_synthesis_b1() {
    let (a, d) = analysis(&[4.0, 6.0, 10.0, 12.0], &HAAR);
    assert_close(&synthesis(&a, &[0.0, 0.0], &HAAR), &[5.0, 5.0, 11.0, 11.0], 1e-4, "low branch alone (B1b)");
    assert_close(&synthesis(&[0.0, 0.0], &d, &HAAR), &[-1.0, 1.0, -1.0, 1.0], 1e-4, "high branch alone (B1b)");
    // Perfect reconstruction for every wavelet.
    let x: Vec<f32> = (0..32).map(|i| (0.3 * i as f32).sin() + if i > 20 { 1.0 } else { 0.0 }).collect();
    for (name, h) in [("Haar", &HAAR[..]), ("db2", &DB2[..]), ("db3", &DB3[..]), ("db4", &DB4[..])] {
        let (a, d) = analysis(&x, h);
        assert_close(&synthesis(&a, &d, h), &x, 1e-4, &format!("{name}: analysis then synthesis"));
        let ex: f32 = x.iter().map(|v| v * v).sum();
        let ec: f32 = a.iter().chain(&d).map(|v| v * v).sum();
        assert!((ex - ec).abs() < 1e-3 * ex, "{name}: Σx² = {ex} but Σa² + Σd² = {ec} (orthogonal: they must agree)");
    }
}

#[test]
fn c5_cascade() {
    // Haar cascade = the orthonormal Haar transform of the last unit (board example).
    let c = dwt(&[4.0, 6.0, 10.0, 12.0, 8.0, 6.0, 5.0, 5.0], &HAAR, 3);
    let r = std::f32::consts::SQRT_2;
    assert_close(&c, &[19.799, 2.828, -6.0, 2.0, -r, -r, r, 0.0], 1e-3, "3-level Haar DWT of the board example");
    let x: Vec<f32> = (0..64).map(|i| ((i * 13) % 17) as f32 / 4.0).collect();
    for levels in [1, 3, 5] {
        let back = idwt(&dwt(&x, &DB4, levels), &DB4, levels);
        assert_close(&back, &x, 1e-3, &format!("db4, {levels} levels: dwt then idwt"));
    }
    // A quadratic has zero db3 details away from the periodic border.
    let q: Vec<f32> = (0..64).map(|i| 0.01 * (i as f32 - 20.0).powi(2)).collect();
    let c = dwt(&q, &DB3, 1);
    for k in 0..28 {
        assert!(c[32 + k].abs() < 1e-3, "db3, quadratic: level-1 detail {k} is {}, expected 0 (3 vanishing moments)", c[32 + k]);
    }
}
