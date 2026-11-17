// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day11 compressed_sensing::

use day11::compressed_sensing::*;
use imgcourse::Rng;

/// The 2 × 4 matrix and the data of R1: y = (3, 6).
fn r1_matrix() -> Vec<Vec<f64>> {
    vec![vec![1.0, 1.0, 1.0, 1.0], vec![1.0, -1.0, 2.0, 0.0]]
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

#[test]
fn r1_min_norm_c1d() {
    let x = min_norm(&r1_matrix(), &[3.0, 6.0]);
    let expected = [1.2, -0.6, 2.1, 0.3];
    for (g, e) in x.iter().zip(expected) {
        assert!(close(*g, e, 1e-9), "C1(d): got {x:?}, expected {expected:?}");
    }
}

#[test]
fn r2_coherence_c8() {
    let mu = coherence(&r1_matrix());
    assert!(close(mu, 3.0 / 10f64.sqrt(), 1e-9), "R8: got {mu}, expected 3/√10 = 0.949 (columns 1 and 3)");
    let identity = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
    assert!(close(coherence(&identity), 0.0, 1e-12), "orthogonal columns have coherence 0");
}

#[test]
fn r3_fold_over_c3() {
    let mut x = vec![0.0; 16];
    x[3] = 1.0;
    let even: Vec<bool> = (0..16).map(|k| k % 2 == 0).collect();
    let z = zero_filled(&x, &even);
    for (n, v) in z.iter().enumerate() {
        let e = if n == 3 || n == 11 { 0.5 } else { 0.0 };
        assert!(close(*v, e, 1e-5), "C3(a), every 2nd coefficient: sample {n}: got {v}, expected {e}");
    }
    let fourth: Vec<bool> = (0..16).map(|k| k % 4 == 0).collect();
    let z = zero_filled(&x, &fourth);
    for (n, v) in z.iter().enumerate() {
        let e = if n % 4 == 3 { 0.25 } else { 0.0 };
        assert!(close(*v, e, 1e-5), "C3(b), every 4th coefficient: sample {n}: got {v}, expected {e}");
    }
    let all = vec![true; 16];
    let z = zero_filled(&x, &all);
    assert!(close(z[3], 1.0, 1e-5), "all coefficients kept: the signal itself, got {z:?}");
}

#[test]
fn r4_recover_the_sparsest_c1() {
    let x = recover(&r1_matrix(), &[3.0, 6.0], 2000);
    let expected = [0.0, 0.0, 3.0, 0.0];
    for (g, e) in x.iter().zip(expected) {
        assert!(close(*g, e, 1e-6), "R1: ℓ1 should find the sparsest solution: got {x:?}, expected {expected:?}");
    }
}

#[test]
fn r4_recover_spikes_from_few_measurements() {
    // 40 random measurements of 4 spikes in 80 unknowns (δ = 0.5, ρ = 0.1: far below the boundary).
    let (m, n) = (40, 80);
    let mut rng = Rng::new(7);
    let a: Vec<Vec<f64>> = (0..m).map(|_| (0..n).map(|_| rng.normal() as f64 / (m as f64).sqrt()).collect()).collect();
    let mut x = vec![0.0; n];
    for (j, v) in [(3, 1.0), (27, -2.0), (50, 0.5), (71, 1.5)] {
        x[j] = v;
    }
    let y = mat_vec(&a, &x);
    let xr = recover(&a, &y, 1000);
    for j in 0..n {
        assert!(close(xr[j], x[j], 1e-4), "unknown {j}: got {}, expected {}", xr[j], x[j]);
    }
}

#[test]
fn r5_success_rate_easy_and_hopeless() {
    let easy = success_rate(32, 24, 2, 3, 1);
    assert!(close(easy, 1.0, 1e-12), "N = 32, M = 24, K = 2 (far below the boundary): got {easy}, expected 1");
    let hopeless = success_rate(32, 8, 6, 3, 1);
    assert!(close(hopeless, 0.0, 1e-12), "N = 32, M = 8, K = 6 (far above the boundary): got {hopeless}, expected 0");
}
