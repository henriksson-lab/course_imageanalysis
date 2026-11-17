// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day11 ista::

use day11::ista::*;

/// The matrix and data of B1: A = [[1, 0.5], [0.5, 1]], y = (2, 1), λ = 0.5, τ = 0.4.
fn b1() -> (Vec<Vec<f64>>, Vec<f64>) {
    (vec![vec![1.0, 0.5], vec![0.5, 1.0]], vec![2.0, 1.0])
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// A normalised circular Gaussian blur (σ = 2) of a signal of length n, as a matrix.
fn blur(n: usize) -> Vec<Vec<f64>> {
    let w: Vec<f64> = (0..n).map(|d| {
        let d = d.min(n - d) as f64;
        (-d * d / 8.0).exp()
    }).collect();
    let s: f64 = w.iter().sum();
    (0..n).map(|i| (0..n).map(|j| w[(i + n - j) % n] / s).collect()).collect()
}

#[test]
fn c1_soft_threshold() {
    let got = soft_threshold(&[1.0, 0.8, 0.1, -0.3, -0.2], 0.2);
    let expected = [0.8, 0.6, 0.0, -0.1, 0.0];
    for (g, e) in got.iter().zip(expected) {
        assert!(close(*g, e, 1e-12), "soft threshold at 0.2: got {got:?}, expected {expected:?}");
    }
}

#[test]
fn c1_two_ista_steps_b1() {
    let (a, y) = b1();
    let x1 = ista_step(&a, &y, &[0.0, 0.0], 0.5, 0.4);
    assert!(close(x1[0], 0.8, 1e-9) && close(x1[1], 0.6, 1e-9), "B1(a): x1 got {x1:?}, expected (0.8, 0.6)");
    let x2 = ista_step(&a, &y, &x1, 0.5, 0.4);
    assert!(close(x2[0], 0.96, 1e-9) && close(x2[1], 0.58, 1e-9), "B1(a): x2 got {x2:?}, expected (0.96, 0.58)");
}

#[test]
fn c2_ista_converges_b1() {
    let (a, y) = b1();
    let x3 = ista(&a, &y, 0.5, 0.4, 3);
    assert!(close(x3[0], 1.048, 1e-9) && close(x3[1], 0.506, 1e-9), "B1(b): x3 got {x3:?}, expected (1.048, 0.506)");
    let x = ista(&a, &y, 0.5, 0.4, 300);
    assert!(close(x[0], 1.6, 1e-4) && x[1] == 0.0, "B1(b): after 300 iterations got {x:?}, expected (1.6, 0) with an exact 0");
    let f = objective(&a, &y, &x, 0.5);
    assert!(close(f, 0.9, 1e-6), "B1(d): objective at the minimum: got {f}, expected 0.9");
}

#[test]
fn c2_objective_never_increases() {
    let a = blur(32);
    let y: Vec<f64> = mat_vec(&a, &(0..32).map(|i| if i % 9 == 2 { 1.0 } else { 0.0 }).collect::<Vec<_>>());
    let mut last = f64::INFINITY;
    for k in 0..30 {
        let f = objective(&a, &y, &ista(&a, &y, 0.02, 1.0, k), 0.02);
        assert!(f <= last + 1e-12, "with τ = 1/L the objective must not increase: iteration {k} gave {f}, before {last}");
        last = f;
    }
}

#[test]
fn c3_largest_gain_b3() {
    let (a, _) = b1();
    let l = largest_gain(&a, 100);
    assert!(close(l, 2.25, 1e-6), "B3(a): L of the B1 matrix: got {l}, expected 2.25");
    let l = largest_gain(&blur(32), 100);
    assert!(close(l, 1.0, 1e-6), "B3(b): L of a normalised blur: got {l}, expected 1");
}

#[test]
fn c4_fista_same_minimum_fewer_steps() {
    let (a, y) = b1();
    let x = fista(&a, &y, 0.5, 0.4, 200);
    assert!(close(x[0], 1.6, 1e-4) && close(x[1], 0.0, 1e-9), "FISTA on B1: got {x:?}, expected (1.6, 0)");
    // On a blur, FISTA gets much closer to the minimum in the same number of iterations.
    let a = blur(64);
    let truth: Vec<f64> = (0..64).map(|i| match i { 10 => 1.0, 30 => -0.8, 34 => 0.7, 50 => 0.5, _ => 0.0 }).collect();
    let y = mat_vec(&a, &truth);
    let best = objective(&a, &y, &fista(&a, &y, 0.01, 1.0, 3000), 0.01);
    let gap_ista = objective(&a, &y, &ista(&a, &y, 0.01, 1.0, 100), 0.01) - best;
    let gap_fista = objective(&a, &y, &fista(&a, &y, 0.01, 1.0, 100), 0.01) - best;
    assert!(gap_fista < 0.2 * gap_ista, "after 100 iterations: FISTA gap {gap_fista:e} should be far below the ISTA gap {gap_ista:e}");
}

#[test]
fn c5_omp_b2() {
    let (a, y) = b1();
    let x = omp(&a, &y, 2, 1e-12);
    assert!(close(x[0], 2.0, 1e-9) && x[1] == 0.0, "B2(b): got {x:?}, expected (2, 0) after one atom");
}

/// The 8 orthonormal DCT basis vectors as the columns of an 8 × 8 matrix.
fn dct8() -> Vec<Vec<f64>> {
    let n = 8.0f64;
    (0..8)
        .map(|i| {
            (0..8)
                .map(|k| {
                    let c = if k == 0 { (1.0 / n).sqrt() } else { (2.0 / n).sqrt() };
                    c * (std::f64::consts::PI * (2 * i + 1) as f64 * k as f64 / (2.0 * n)).cos()
                })
                .collect()
        })
        .collect()
}

#[test]
fn c5_omp_masked_dct_b4() {
    // Keep the rows (samples) 0, 2, 3, 6 of the DCT matrix: the masked atoms.
    let d = dct8();
    let a: Vec<Vec<f64>> = [0, 2, 3, 6].iter().map(|&i| d[i].clone()).collect();
    let y = [1.123, 0.217, 0.429, 0.805];
    let x1 = omp(&a, &y, 1, 1e-12);
    assert!(x1[0] != 0.0 && x1.iter().filter(|v| **v != 0.0).count() == 1, "B4(c): the first atom should be atom 0, got {x1:?}");
    assert!(close(x1[0], 1.82, 0.01), "B4(c): first refit a = {}, expected 1.82", x1[0]);
    let x = omp(&a, &y, 2, 1e-12);
    assert!(close(x[0], 2.0, 2e-3) && close(x[3], 1.0, 2e-3), "B4(c): after two atoms got {x:?}, expected a = 2 (atom 0), b = 1 (atom 3)");
}
