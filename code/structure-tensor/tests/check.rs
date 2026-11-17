// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p structure-tensor

use imgcourse::Image;
use structure_tensor::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

/// A bright square (100) on black (0): columns and rows 16 … 47 of a 64 × 64 image.
fn square() -> Image<f32> {
    Image::from_fn(64, 64, |x, y| if (16..48).contains(&x) && (16..48).contains(&y) { 100.0 } else { 0.0 })
}

#[test]
fn c1_tensor_of_gradients_b1_b2() {
    let cases = [
        (vec![[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], [1.0, 0.0, 0.0]), // B1
        (vec![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], [0.5, 0.0, 0.5]), // B2(a)
        (vec![[0.0, 0.0]; 4], [0.0, 0.0, 0.0]),                                  // B2(b)
        (vec![[1.0, 2.0], [3.0, -1.0]], [5.0, -0.5, 2.5]),
    ];
    for (grads, expected) in cases {
        let j = tensor_of_gradients(&grads);
        assert!(j.iter().zip(expected).all(|(a, b)| close(*a, b, 1e-6)), "gradients {grads:?}: got J = {j:?}, expected {expected:?}");
    }
}

#[test]
fn c2_eigen_and_coherence_b3() {
    let (mu1, mu2, theta) = eigen_tensor([5.0, 2.0, 2.0]);
    assert!(close(mu1, 6.0, 1e-5) && close(mu2, 1.0, 1e-5), "B3(a): got μ1 = {mu1}, μ2 = {mu2}, expected 6 and 1");
    let c = coherence(mu1, mu2);
    assert!(close(c, 25.0 / 49.0, 1e-4), "B3(a): coherence {c}, expected (5/7)² = 0.51");
    assert!(close(theta.to_degrees(), 26.57, 0.01), "B3(b): θ = {:.2}°, expected 26.57°", theta.to_degrees());
    let (mu1, mu2, theta) = eigen_tensor([0.0, 0.0, 3.0]); // vertical gradients: a horizontal edge
    assert!(close(mu1, 3.0, 1e-6) && close(mu2, 0.0, 1e-6) && close(theta.abs().to_degrees(), 90.0, 1e-3), "J = [0 0; 0 3]: got μ = ({mu1}, {mu2}), θ = {:.1}°, expected (3, 0), ±90°", theta.to_degrees());
    for (mu1, mu2, expected) in [(1.0, 0.0, 1.0), (0.5, 0.5, 0.0), (0.0, 0.0, 0.0)] {
        let c = coherence(mu1, mu2);
        assert!(close(c, expected, 1e-6), "B1/B2: coherence of μ = ({mu1}, {mu2}) is {c}, expected {expected}");
    }
}

#[test]
fn c3_harris_b4() {
    let cases = [([10.0, 0.0, 0.1], -4.10), ([10.0, 0.0, 8.0], 63.8), ([0.1, 0.0, 0.1], 0.008)];
    for (j, expected) in cases {
        let r = harris(j, 0.05);
        assert!(close(r, expected, 0.005), "B4, J = {j:?}, k = 0.05: got R = {r}, expected {expected}");
    }
}

#[test]
fn c4_structure_tensor_edge_corner_flat() {
    let j = structure_tensor(&square(), 1.0, 3.0);
    // Middle of the left edge: one large eigenvalue, gradient horizontal.
    let (mu1, mu2, theta) = eigen_tensor(j[(16, 32)]);
    assert!(mu1 > 100.0 && coherence(mu1, mu2) > 0.99, "edge (16, 32): μ = ({mu1}, {mu2}), coherence {}, expected one large eigenvalue", coherence(mu1, mu2));
    assert!(close(theta.to_degrees(), 0.0, 1.0), "edge (16, 32): dominant gradient at {:.1}°, expected 0° (across the vertical edge)", theta.to_degrees());
    // Corner: two large eigenvalues.
    let (mu1, mu2, _) = eigen_tensor(j[(16, 16)]);
    assert!(mu2 > 0.3 * mu1 && coherence(mu1, mu2) < 0.2, "corner (16, 16): μ = ({mu1}, {mu2}), expected two large eigenvalues");
    // Flat inside.
    let (mu1, _, _) = eigen_tensor(j[(32, 32)]);
    assert!(mu1 < 1e-3, "inside the square (32, 32): μ1 = {mu1}, expected about 0");
    // Without a window, J has rank 1 at every pixel: μ2 = 0 even at the corner.
    let j0 = structure_tensor(&square(), 1.0, 0.0);
    let (mu1, mu2, _) = eigen_tensor(j0[(16, 16)]);
    assert!(mu2.abs() < 1e-3 * mu1, "ρ = 0, corner (16, 16): μ = ({mu1}, {mu2}), expected μ2 = 0");
}

#[test]
fn c5_four_corners_of_a_square() {
    let j = structure_tensor(&square(), 1.0, 2.0);
    let r = j.map(|t| harris(t, 0.05));
    let found = corners(&r, 0.1 * r.max());
    assert_eq!(found.len(), 4, "the square has 4 corners, found {found:?}");
    for (cx, cy) in [(16, 16), (47, 16), (16, 47), (47, 47)] {
        let near = found.iter().any(|&(x, y)| x.abs_diff(cx) <= 2 && y.abs_diff(cy) <= 2);
        assert!(near, "no corner found near ({cx}, {cy}): found {found:?}");
    }
}
