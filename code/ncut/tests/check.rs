// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p ncut

use ncut::*;

fn graph(n: usize, edges: &[(usize, usize, f64)]) -> Vec<Vec<f64>> {
    let mut w = vec![vec![0.0; n]; n];
    for &(p, q, v) in edges {
        w[p][q] = v;
        w[q][p] = v;
    }
    w
}

/// B1–B3: the path 1 − 2 − 3 − 4 (nodes 0..3 in the code), weights 1.
fn path4() -> Vec<Vec<f64>> {
    graph(4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
}

/// B4: two triangles 1, 2, 3 and 4, 5, 6, the edge 3 − 4 of weight 0.1, and node 7 on node 6
/// with weight 0.05 (nodes 0..6 in the code).
fn triangles() -> Vec<Vec<f64>> {
    graph(7, &[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0), (3, 4, 1.0), (4, 5, 1.0), (3, 5, 1.0), (2, 3, 0.1), (5, 6, 0.05)])
}

fn set(n: usize, nodes: &[usize]) -> Vec<bool> {
    (0..n).map(|p| nodes.contains(&p)).collect()
}

#[test]
fn c1_affinities() {
    let w = affinities(&[[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], 1.0);
    assert_eq!(w[0][0], 0.0, "no node is joined to itself");
    assert!((w[0][1] - (-0.5f64).exp()).abs() < 1e-9, "distance 1, σ = 1: got {}, expected e^(−1/2)", w[0][1]);
    assert!((w[2][0] - (-2.0f64).exp()).abs() < 1e-9, "distance 2, σ = 1: got {}, expected e^(−2)", w[2][0]);
    assert_eq!(w[1][2], w[2][1], "the affinities must be symmetric");
}

#[test]
fn c2_laplacian_b1() {
    let (d, l) = laplacian(&path4());
    assert_eq!(d, vec![1.0, 2.0, 2.0, 1.0], "B1(a): degrees");
    let expected = [[1.0, -1.0, 0.0, 0.0], [-1.0, 2.0, -1.0, 0.0], [0.0, -1.0, 2.0, -1.0], [0.0, 0.0, -1.0, 1.0]];
    for p in 0..4 {
        for q in 0..4 {
            assert_eq!(l[p][q], expected[p][q], "B1(a): L row {}, column {}", p + 1, q + 1);
        }
    }
}

#[test]
fn c2_quadratic_form_b2() {
    let (_, l) = laplacian(&path4());
    let a = quadratic_form(&l, &[1.0, 1.0, -1.0, -1.0]);
    let b = quadratic_form(&l, &[1.0, -1.0, 1.0, -1.0]);
    assert!((a - 4.0).abs() < 1e-9, "B2: xᵀLx for (1, 1, −1, −1): got {a}, expected 4");
    assert!((b - 12.0).abs() < 1e-9, "B2: xᵀLx for (1, −1, 1, −1): got {b}, expected 12");
    assert!(quadratic_form(&l, &[1.0; 4]).abs() < 1e-9, "B1(b): xᵀLx = 0 for the constant vector");
}

#[test]
fn c3_ncut_b3_b4() {
    let (cut, nc) = ncut(&path4(), &set(4, &[0, 1]));
    assert!((cut - 1.0).abs() < 1e-9 && (nc - 2.0 / 3.0).abs() < 1e-9, "B3(b): {{1, 2}} | {{3, 4}}: cut {cut}, Ncut {nc}, expected 1 and 0.667");
    let (_, nc) = ncut(&path4(), &set(4, &[0]));
    assert!((nc - 1.2).abs() < 1e-9, "B3(b): {{1}} | {{2, 3, 4}}: Ncut {nc}, expected 1.2");
    let w = triangles();
    for (nodes, cut, nc) in [(vec![6], 0.05, 1.004), (vec![0, 1, 2], 0.1, 0.0325), (vec![5, 6], 2.0, 1.149)] {
        let (c, n) = ncut(&w, &set(7, &nodes));
        assert!((c - cut).abs() < 1e-9, "B4(b, c): cut of {nodes:?} (from 0): got {c}, expected {cut}");
        assert!((n - nc).abs() < 1e-3, "B4(c): Ncut of {nodes:?} (from 0): got {n:.4}, expected {nc}");
    }
}

#[test]
fn c4_fiedler_path_b3() {
    let (mu, x) = fiedler(&path4(), 200);
    assert!((mu - 0.5).abs() < 1e-6, "B3(a): μ₂ = {mu}, expected 1/2");
    let r: Vec<f64> = x.iter().map(|v| v / x[0]).collect();
    let expected = [1.0, 0.5, -0.5, -1.0];
    for p in 0..4 {
        assert!((r[p] - expected[p]).abs() < 1e-5, "B3(a): x should be proportional to (1, ½, −½, −1), got {x:?}");
    }
}

#[test]
fn c4_fiedler_triangles_b4() {
    let (mu, x) = fiedler(&triangles(), 500);
    assert!((mu - 0.031).abs() < 0.001, "B4: μ₂ = {mu}, expected about 0.031");
    let first = x[0].signum();
    assert!((0..3).all(|p| x[p].signum() == first) && (3..7).all(|p| x[p].signum() == -first), "B4: the sign should split {{1, 2, 3}} from {{4, 5, 6, 7}}, got {x:?}");
}

#[test]
fn c5_sweep_b4() {
    let w = triangles();
    let (_, x) = fiedler(&w, 500);
    let (split, nc) = sweep(&w, &x);
    let a: Vec<usize> = (0..7).filter(|&p| split[p]).collect();
    assert!(a == vec![0, 1, 2] || a == vec![3, 4, 5, 6], "B4: the best sweep split separates the triangles, got {a:?}");
    assert!((nc - 0.0325).abs() < 1e-3, "B4: its Ncut {nc}, expected 0.0325");
}
