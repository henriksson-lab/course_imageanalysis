// Checks for the coding exercises, using the numbers of the by-hand exercises.
// Run: cargo test -p regularise

use imgcourse::Rng;
use regularise::*;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

fn mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| row.iter().zip(x).map(|(p, q)| p * q).sum()).collect()
}

#[test]
fn c1_blur_matrix() {
    let a = blur_matrix(20, 1.5);
    assert_eq!((a.len(), a[0].len()), (20, 20), "size: got {} × {}, expected 20 × 20", a.len(), a[0].len());
    for (i, row) in a.iter().enumerate() {
        let s: f64 = row.iter().sum();
        assert!(close(s, 1.0, 1e-12), "row {i} sums to {s}, expected 1");
    }
    // in the middle: Gaussian weights, symmetric, zero beyond 3σ (5 samples)
    assert!(close(a[10][11] / a[10][10], (-1.0f64 / 4.5).exp(), 1e-12), "row 10: a[10][11] / a[10][10] = {}, expected exp(−1/(2σ²))", a[10][11] / a[10][10]);
    assert!(close(a[10][9], a[10][11], 1e-15), "row 10 is not symmetric");
    assert!(a[10][15] > 0.0 && a[10][16] == 0.0, "row 10: weights up to 5 samples away (ceil(3σ)), got a[10][15] = {}, a[10][16] = {}", a[10][15], a[10][16]);
}

#[test]
fn c2_two_pixels_deconv_b3() {
    // The two-pixel blur of the deconvolution unit, data (1, 0.5), energy penalty.
    let a = vec![vec![0.75, 0.25], vec![0.25, 0.75]];
    for (lambda, e) in [(1e-12, [1.25, 0.25]), (0.1, [1.039, 0.325]), (1.0, [0.475, 0.275])] {
        let x = tikhonov(&a, &[1.0, 0.5], lambda, Penalty::Energy);
        assert!(close(x[0], e[0], 1e-3) && close(x[1], e[1], 1e-3), "λ = {lambda}: got {x:?}, expected {e:?}");
    }
}

#[test]
fn c2_roughness_keeps_a_ramp() {
    // A ramp has no roughness and fits noise-free data exactly: every λ must return it.
    let a = blur_matrix(30, 2.0);
    let ramp: Vec<f64> = (0..30).map(|i| 0.1 + 0.02 * i as f64).collect();
    let y = mul(&a, &ramp);
    for lambda in [0.01, 1.0, 100.0] {
        let x = tikhonov(&a, &y, lambda, Penalty::Roughness);
        for i in 0..30 {
            assert!(close(x[i], ramp[i], 1e-6), "roughness, λ = {lambda}, sample {i}: got {}, expected the ramp {}", x[i], ramp[i]);
        }
        let e = tikhonov(&a, &y, lambda, Penalty::Energy);
        assert!(e.iter().sum::<f64>() < ramp.iter().sum::<f64>(), "energy, λ = {lambda}: should shrink the ramp");
    }
}

#[test]
fn c3_factors_b1_b2() {
    for (s, l, e) in [(1.0, 0.01, 0.990), (0.5, 0.01, 0.962), (0.1, 0.01, 0.5), (1.0, 0.1, 0.909), (0.5, 0.1, 0.714), (0.1, 0.1, 0.091)] {
        let f = tikhonov_factor(s, l);
        assert!(close(f, e, 1e-3), "Tikhonov, s = {s}, λ = {l}: got {f}, expected {e}");
    }
    for (s, k, e) in [(0.5, 1, 0.25), (0.5, 4, 0.684), (0.5, 16, 0.990), (0.1, 16, 0.149), (0.1, 100, 0.634)] {
        let f = landweber_factor(s, k);
        assert!(close(f, e, 1e-3), "Landweber, s = {s}, k = {k}: got {f}, expected {e}");
    }
}

#[test]
fn c4_landweber_b2() {
    // A diagonal blur with gains 1, 0.5, 0.1 and the object (1, 1, 1): after k steps each
    // component holds its filter factor.
    let a = vec![vec![1.0, 0.0, 0.0], vec![0.0, 0.5, 0.0], vec![0.0, 0.0, 0.1]];
    let y = [1.0, 0.5, 0.1];
    for k in [1, 4, 16, 100] {
        let x = landweber(&a, &y, k, 1.0);
        for (i, s) in [1.0, 0.5, 0.1].into_iter().enumerate() {
            let e = landweber_factor(s, k);
            assert!(close(x[i], e, 1e-9), "k = {k}, component with s = {s}: got {}, expected {e}", x[i]);
        }
    }
}

#[test]
fn c5_rms_residual() {
    let a = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
    let r = rms_residual(&a, &[1.0, 1.0], &[0.0, 0.0]);
    assert!(close(r, (5.0f64 / 2.0).sqrt(), 1e-12), "residual of (1, 2) against 0: got {r}, expected sqrt(5/2)");
}

#[test]
fn c5_discrepancy() {
    let n = 48;
    let a = blur_matrix(n, 2.0);
    let truth: Vec<f64> = (0..n).map(|i| if (12..30).contains(&i) { 0.8 } else { 0.2 }).collect();
    let mut rng = Rng::new(5);
    let sigma = 0.02;
    let y: Vec<f64> = mul(&a, &truth).iter().map(|v| v + sigma * rng.normal() as f64).collect();
    let lambda = discrepancy_lambda(&a, &y, sigma, Penalty::Roughness);
    let r = rms_residual(&a, &tikhonov(&a, &y, lambda, Penalty::Roughness), &y);
    assert!(close(r, sigma, 1e-4), "at the discrepancy λ = {lambda:e} the RMS residual is {r}, expected σ = {sigma}");
    let below = rms_residual(&a, &tikhonov(&a, &y, lambda / 10.0, Penalty::Roughness), &y);
    assert!(below < sigma, "at λ / 10 the residual should be below σ, got {below}");
}
