// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p snake-matrix

use imgcourse::Image;
use snake_matrix::*;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

/// Mode k of a closed snake of n beads: the wave cos(2πkj/n).
fn wave(n: usize, k: usize) -> Vec<f64> {
    (0..n).map(|j| (2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64).cos()).collect()
}

/// The factor by which a step multiplies the wave (the step must give a multiple of it).
fn factor(before: &[f64], after: &[f64]) -> f64 {
    let i = (0..before.len()).max_by(|&i, &j| before[i].abs().total_cmp(&before[j].abs())).unwrap();
    let f = after[i] / before[i];
    for j in 0..before.len() {
        assert!((after[j] - f * before[j]).abs() < 1e-6, "the step should multiply the wave by one number: {before:?} became {after:?}");
    }
    f
}

#[test]
fn c1_matrix_b1() {
    let a = snake_matrix(6, 1.0, 0.0);
    let expected = [
        [2., -1., 0., 0., 0., -1.],
        [-1., 2., -1., 0., 0., 0.],
        [0., -1., 2., -1., 0., 0.],
        [0., 0., -1., 2., -1., 0.],
        [0., 0., 0., -1., 2., -1.],
        [-1., 0., 0., 0., -1., 2.],
    ];
    for i in 0..6 {
        for j in 0..6 {
            assert!(close(a[i][j], expected[i][j]), "row {i}, column {j}: got {}, expected {}", a[i][j], expected[i][j]);
        }
    }
    let alternating = [1., -1., 1., -1., 1., -1.];
    for i in 0..6 {
        let row_sum: f64 = a[i].iter().sum();
        assert!(close(row_sum, 0.0), "B1(b): row {i} sums to {row_sum}, expected 0");
        let av: f64 = (0..6).map(|j| a[i][j] * alternating[j]).sum();
        assert!(close(av, 4.0 * alternating[i]), "B1(c): (A v)[{i}] = {av}, expected {}", 4.0 * alternating[i]);
    }
}

#[test]
fn c1_matrix_with_beta() {
    let a = snake_matrix(6, 1.0, 1.0);
    let expected = [8., -5., 1., 0., 1., -5.];
    for j in 0..6 {
        assert!(close(a[0][j], expected[j]), "α = β = 1, row 0: got {:?}, expected {expected:?}", a[0]);
    }
}

#[test]
fn c2_explicit_factors_b3() {
    let a = snake_matrix(6, 1.0, 0.0);
    let zero = vec![0.0; 6];
    for (tau, expected) in [(0.1, [0.9, 0.7, 0.6]), (0.5, [0.5, -0.5, -1.0]), (1.0, [0.0, -2.0, -3.0])] {
        for k in 1..=3 {
            let w = wave(6, k);
            let f = factor(&w, &explicit_step(&a, &w, &zero, tau));
            assert!(close(f, expected[k - 1]), "τ = {tau}, mode {k}: factor {f}, expected {}", expected[k - 1]);
        }
    }
}

#[test]
fn c2_image_force_is_added() {
    let a = snake_matrix(5, 0.3, 0.1);
    let x = vec![1.0; 5]; // a shift: no internal force
    let f = vec![0.5, -1.0, 0.0, 2.0, 1.0];
    let new = explicit_step(&a, &x, &f, 2.0);
    for i in 0..5 {
        assert!(close(new[i], 1.0 + 2.0 * f[i]), "bead {i}: got {}, expected {}", new[i], 1.0 + 2.0 * f[i]);
    }
}

#[test]
fn c3_inverse() {
    let a = snake_matrix(7, 0.1, 0.05);
    let inv = semi_implicit_inverse(&a, 2.0);
    for i in 0..7 {
        for j in 0..7 {
            let p: f64 = (0..7).map(|k| (a[i][k] + if i == k { 0.5 } else { 0.0 }) * inv[k][j]).sum();
            let e = if i == j { 1.0 } else { 0.0 };
            assert!((p - e).abs() < 1e-9, "(A + γI) times your inverse, row {i}, column {j}: got {p}, expected {e}");
        }
    }
}

#[test]
fn c4_semi_implicit_factors_b4() {
    let a = snake_matrix(6, 1.0, 0.0);
    let zero = vec![0.0; 6];
    let inv = semi_implicit_inverse(&a, 1.0);
    for (k, expected) in [(1, 0.5), (2, 0.25), (3, 0.2)] {
        let w = wave(6, k);
        let f = factor(&w, &semi_implicit_step(&inv, &w, &zero, 1.0));
        assert!(close(f, expected), "τ = 1, mode {k}: factor {f}, expected {expected}");
    }
    // B4(b): a huge step leaves only the mean position.
    let inv = semi_implicit_inverse(&a, 1e6);
    let x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0];
    let new = semi_implicit_step(&inv, &x, &zero, 1e6);
    for v in &new {
        assert!((v - 23.0 / 6.0).abs() < 1e-3, "huge τ: got {new:?}, expected every bead at the mean 3.833");
    }
}

#[test]
fn c5_gvf_b6() {
    let g = Image::from_rows(&[[0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0]]);
    let zero = Image::new(7, 1, 0.0);
    let after = |iterations: usize| gvf(&g, &zero, 0.25, 1.0, iterations).0;
    let expected1 = [0.0, 0.25, 0.5, 0.0, -0.5, -0.25, 0.0];
    let expected2 = [0.0625, 0.25, 0.8125, 0.0, -0.8125, -0.25, -0.0625];
    for (iterations, expected) in [(1, expected1), (2, expected2)] {
        let v = after(iterations);
        for i in 0..7 {
            assert!((v[(i, 0)] - expected[i]).abs() < 1e-5, "B6, after {iterations} updates, sample {i}: got {}, expected {}", v[(i, 0)], expected[i]);
        }
    }
    let (_, vy) = gvf(&g, &zero, 0.25, 1.0, 2);
    assert!(vy.pixels().iter().all(|v| v.abs() < 1e-7), "the y component should stay 0");
}
