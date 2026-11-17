// Checks for the coding exercises, using the numbers of the lecture and the pen-and-paper
// exercises. Run: cargo test -p svm kernel_trick::

use imgcourse::Rng;
use svm::kernel_trick::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[test]
fn c1_quadratic_map_lecture_and_b1() {
    // Lecture: x = (1, 2), z = (3, 1): φ(x) · φ(z) = 25 = (x · z)².
    let d = dot3(quadratic_map([1.0, 2.0]), quadratic_map([3.0, 1.0]));
    assert!(close(d, 25.0, 1e-4), "lecture: φ(1, 2) · φ(3, 1) = {d}, expected 25");
    // B1(a): x = (2, −1), z = (1, 3): 1.
    let (px, pz) = (quadratic_map([2.0, -1.0]), quadratic_map([1.0, 3.0]));
    let expected = [4.0, -2.0 * 2.0f32.sqrt(), 1.0];
    for i in 0..3 {
        assert!(close(px[i], expected[i], 1e-4), "B1(a): φ(2, −1) = {px:?}, expected (4, −2√2, 1)");
    }
    assert!(close(dot3(px, pz), 1.0, 1e-4), "B1(a): φ(x) · φ(z) = {}, expected 1", dot3(px, pz));
}

#[test]
fn c2_kernel_functions() {
    let quad = Kernel::Polynomial { c: 0.0, degree: 2 };
    let k = kernel(quad, &[2.0, -1.0], &[1.0, 3.0]);
    assert!(close(k, 1.0, 1e-4), "B1(a): (x · z)² = {k}, expected 1");
    let k = kernel(Kernel::Polynomial { c: 1.0, degree: 2 }, &[1.0, 2.0], &[3.0, 1.0]);
    assert!(close(k, 36.0, 1e-4), "B1(b): (x · z + 1)² = {k}, expected 36");
    let k = kernel(Kernel::Linear, &[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
    assert!(close(k, 32.0, 1e-4), "linear, 3 features: got {k}, expected 32");
    // B4: neighbouring points 1 apart.
    for (gamma, e) in [(0.01, (-0.01f32).exp()), (1.0, (-1.0f32).exp()), (100.0, 0.0)] {
        let k = kernel(Kernel::Rbf { gamma }, &[0.0, 0.0], &[1.0, 0.0]);
        assert!(close(k, e, 1e-4), "B4: RBF with γ = {gamma} at distance 1: got {k}, expected {e}");
    }
    let k = kernel(Kernel::Rbf { gamma: 5.0 }, &[0.3, 0.7], &[0.3, 0.7]);
    assert!(close(k, 1.0, 1e-6), "RBF of a point with itself: got {k}, expected 1");
}

#[test]
fn c3_decision_b3() {
    // B3: support vectors (0, 0) with α y = +1 and (2, 0) with α y = −1, γ = 1, b = 0.
    let svs = vec![vec![0.0, 0.0], vec![2.0, 0.0]];
    let (labels, alpha) = ([1.0, -1.0], [1.0, 1.0]);
    let rbf = Kernel::Rbf { gamma: 1.0 };
    let cases = [([0.0, 0.0], 0.982), ([0.5, 0.0], 0.673), ([1.0, 0.0], 0.0), ([0.0, 1.0], 0.361), ([3.0, 0.0], -0.368), ([-1.0, 0.0], 0.368)];
    for (x, e) in cases {
        let f = decision(&svs, &labels, &alpha, 0.0, rbf, &x);
        assert!(close(f, e, 2e-3), "B3: f{x:?} = {f}, expected {e}");
    }
    // The lecture's "two bumps": at (0.5, 0) with γ = 0.1 and γ = 10.
    for (gamma, e) in [(0.1, 0.177), (10.0, 0.082)] {
        let f = decision(&svs, &labels, &alpha, 0.0, Kernel::Rbf { gamma }, &[0.5, 0.0]);
        assert!(close(f, e, 2e-3), "two bumps, γ = {gamma}: f(0.5, 0) = {f}, expected {e}");
    }
}

#[test]
fn c4_smo_two_points_b7() {
    let points = vec![vec![3.0, 1.0], vec![1.0, 2.0]];
    let labels = [1.0, -1.0];
    let (alpha, b) = train_smo(&points, &labels, Kernel::Linear, 1000.0, 1e-4, 1000);
    assert!(close(alpha[0], 0.4, 1e-3) && close(alpha[1], 0.4, 1e-3), "B7: α = {alpha:?}, expected [0.4, 0.4]");
    assert!(close(b, -1.0, 1e-3), "B7: b = {b}, expected −1");
}

#[test]
fn c4_smo_lecture_example() {
    // The previous unit's example and B9 there: α = (0.25, 0.25, 0.5, 0), w = (1, 0), b = −1.
    let points = vec![vec![0.0, 0.0], vec![0.0, 2.0], vec![2.0, 1.0], vec![3.0, 0.0]];
    let labels = [-1.0, -1.0, 1.0, 1.0];
    let (alpha, b) = train_smo(&points, &labels, Kernel::Linear, 1000.0, 1e-4, 1000);
    let expected = [0.25, 0.25, 0.5, 0.0];
    for i in 0..4 {
        assert!(close(alpha[i], expected[i], 1e-3), "α = {alpha:?}, expected {expected:?}");
    }
    assert!(close(b, -1.0, 1e-3), "b = {b}, expected −1");
    let total: f32 = (0..4).map(|i| alpha[i] * labels[i]).sum();
    assert!(close(total, 0.0, 1e-5), "Σ α y = {total}, expected 0");
    let f = decision(&points, &labels, &alpha, b, Kernel::Linear, &[1.0, 5.0]);
    assert!(close(f, 0.0, 1e-3), "f(1, 5) = {f}, expected 0: the line is x1 = 1");
}

#[test]
fn c4_smo_xor_needs_degree_2() {
    // B2: XOR at (±1, ±1), with a little noise: linear fails, (x · z + 1)² separates.
    let mut rng = Rng::new(1);
    let (mut points, mut labels) = (Vec::new(), Vec::new());
    for (cx, cy, y) in [(1.0, 1.0, -1.0), (-1.0, -1.0, -1.0), (1.0, -1.0, 1.0), (-1.0, 1.0, 1.0)] {
        for _ in 0..10 {
            points.push(vec![cx + 0.2 * rng.normal(), cy + 0.2 * rng.normal()]);
            labels.push(y);
        }
    }
    let errors = |k: Kernel| {
        let (alpha, b) = train_smo(&points, &labels, k, 100.0, 1e-3, 100000);
        (0..points.len()).filter(|&i| labels[i] * decision(&points, &labels, &alpha, b, k, &points[i]) <= 0.0).count()
    };
    let e = errors(Kernel::Polynomial { c: 1.0, degree: 2 });
    assert_eq!(e, 0, "XOR with (x · z + 1)²: expected 0 training errors, got {e}");
    let e = errors(Kernel::Linear);
    assert!(e >= 10, "XOR with the linear kernel function: expected many errors, got {e}");
}
