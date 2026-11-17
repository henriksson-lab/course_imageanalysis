// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p unmixing endmembers::

use unmixing::endmembers::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn r1_nnls_b1() {
    let m = vec![vec![1.0, 0.3], vec![0.2, 1.0]];
    let a = nnls(&m, &[0.1, 0.9]);
    assert!(close(a[0], 0.0, 1e-6) && close(a[1], 0.853, 1e-3), "B1(c): got {a:?}, expected (0, 0.853)");
    // KKT (B7): w = Mᵀ(y − M a) must be ≤ 0 where a = 0
    let r = [0.1 - 0.3 * a[1], 0.9 - a[1]];
    let w0 = m[0][0] * r[0] + m[1][0] * r[1];
    assert!(close(w0, -0.147, 1e-3), "B7: w1 = {w0}, expected −0.147");
}

#[test]
fn r1_nnls_inside_is_ls() {
    // a pixel inside the cone: NNLS = ordinary least squares
    let m = vec![vec![1.0, 0.2], vec![0.5, 0.5], vec![0.1, 1.0]];
    let a = nnls(&m, &[1.3, 0.9, 1.2]);
    assert!(close(a[0], 1.029, 1e-3) && close(a[1], 1.042, 1e-3), "a pixel inside the cone: got {a:?}, expected (1.029, 1.042)");
    // a pixel pointing away from every spectrum: all zero
    let a = nnls(&m, &[-1.0, -1.0, -1.0]);
    assert!(a.iter().all(|&v| v == 0.0), "y = (−1, −1, −1): got {a:?}, expected all 0");
}

#[test]
fn r2_fractions_b2() {
    let e = [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]];
    let a = fractions([1.0, 1.0], e);
    let x = [0.417, 0.25, 0.333];
    assert!((0..3).all(|k| close(a[k], x[k], 1e-3)), "B2(a): got {a:?}, expected {x:?}");
    let a = fractions([3.0, 1.5], e);
    let x = [-0.25, 0.75, 0.5];
    assert!((0..3).all(|k| close(a[k], x[k], 1e-3)), "B2(b): got {a:?}, expected {x:?}");
}

#[test]
fn r3_singular_values() {
    // Y = (3 0 0 ; 0 0 4 ; 0 0 0): singular values 4, 3, 0
    let y = vec![vec![3.0, 0.0, 0.0], vec![0.0, 0.0, 4.0], vec![0.0, 0.0, 0.0]];
    let s = singular_values(&y);
    assert!(close(s[0], 4.0, 1e-4) && close(s[1], 3.0, 1e-4) && close(s[2], 0.0, 1e-3), "got {s:?}, expected (4, 3, 0)");
    // pixels made of two spectra: exactly two non-zero singular values
    let (m1, m2) = ([1.0, 0.5, 0.2, 0.0], [0.0, 0.3, 0.8, 1.0]);
    let amounts = [(1.0, 0.0), (0.0, 2.0), (0.5, 0.5), (1.5, 0.2), (0.3, 1.1)];
    let y: Vec<Vec<f32>> = (0..4).map(|c| amounts.iter().map(|&(a, b)| a * m1[c] + b * m2[c]).collect()).collect();
    let s = singular_values(&y);
    assert!(s[1] > 0.1 && s[2] < 1e-3 && s[3] < 1e-3, "two components: got {s:?}, expected two values above 0");
}

#[test]
fn r4_vertices_pure_pixels() {
    // mixtures of three spectra, with one pure pixel of each among them (indices 3, 6, 8)
    let spectra = [[1.0, 0.4, 0.1, 0.0], [0.1, 0.8, 0.6, 0.1], [0.0, 0.1, 0.5, 1.0]];
    let amounts = [[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.3, 0.4], [1.0, 0.0, 0.0], [0.6, 0.2, 0.2], [0.1, 0.6, 0.3], [0.0, 1.0, 0.0], [0.4, 0.1, 0.5], [0.0, 0.0, 1.0]];
    let pixels: Vec<Vec<f32>> = amounts.iter().map(|a| (0..4).map(|c| (0..3).map(|k| a[k] * spectra[k][c]).sum()).collect()).collect();
    let mut found = find_vertices(&pixels, 3);
    found.sort();
    assert!(found == vec![3, 6, 8], "the pure pixels are 3, 6 and 8: got {found:?}");
}

#[test]
fn r5_update_h_b8() {
    let y = vec![vec![3.0, 1.0], vec![1.0, 3.0]];
    let w = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
    let mut h = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
    update_h(&y, &w, &mut h);
    let e = [[1.556, 1.111], [1.111, 1.556]];
    for i in 0..2 {
        for j in 0..2 {
            assert!(close(h[i][j], e[i][j], 1e-3), "B8, H at ({}, {}): got {}, expected {}", i + 1, j + 1, h[i][j], e[i][j]);
        }
    }
}

#[test]
fn r5_updates_lower_the_error() {
    let y = vec![vec![3.0, 1.0, 2.0], vec![1.0, 3.0, 2.0], vec![0.5, 0.2, 1.0]];
    let mut w = vec![vec![1.0, 0.5], vec![0.5, 1.0], vec![0.3, 0.3]];
    let mut h = vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]];
    let err = |w: &Matrix, h: &Matrix| -> f32 {
        (0..3).flat_map(|c| (0..3).map(move |q| (c, q))).map(|(c, q)| (y[c][q] - (0..2).map(|j| w[c][j] * h[j][q]).sum::<f32>()).powi(2)).sum()
    };
    let mut last = err(&w, &h);
    for step in 0..20 {
        update_h(&y, &w, &mut h);
        update_w(&y, &mut w, &h);
        let e = err(&w, &h);
        assert!(e <= last + 1e-5, "step {step}: the error rose from {last} to {e}");
        last = e;
    }
    assert!(w.iter().flatten().chain(h.iter().flatten()).all(|&v| v >= 0.0), "W and H must stay non-negative");
}
