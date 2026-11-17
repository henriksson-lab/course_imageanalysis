// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day19 ripley_k::

use imgcourse::Rng;
use day19::ripley_k::*;
use std::f64::consts::PI;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// The five points of B1 in a 10 × 10 µm window.
const B1: [Point; 5] = [(2.0, 2.0), (3.0, 2.0), (4.0, 4.0), (7.0, 5.0), (8.0, 7.0)];

#[test]
fn c1_translation_weight_b4() {
    let w = translation_weight(20.0, 10.0, 100.0, 50.0);
    assert!(close(w, 1.5625, 1e-12), "100 × 50 window, pair (20, 10): got {w}, expected 1.5625");
    let w = translation_weight(-20.0, 10.0, 100.0, 50.0);
    assert!(close(w, 1.5625, 1e-12), "the sign of the difference does not matter: got {w}, expected 1.5625");
    let w = translation_weight(20.0, 10.0, 1000.0, 500.0);
    assert!(close(w, 500_000.0 / (980.0 * 490.0), 1e-12), "1000 × 500 window: got {w:.4}, expected 1.0412");
}

#[test]
fn c2_k_and_l_b1() {
    for (r, k_expected, l_expected) in [(1.0, 10.0, 0.784), (2.0, 10.0, -0.216), (3.0, 40.0, 0.568), (4.0, 50.0, -0.011)] {
        let k = k_hat(&B1, 10.0, 10.0, r, false);
        assert!(close(k, k_expected, 1e-9), "B1(b), r = {r}: K = {k}, expected {k_expected}");
        let l = l_minus_r(k, r);
        assert!(close(l, l_expected, 2e-3), "B1(c), r = {r}: L - r = {l:.3}, expected {l_expected}");
    }
}

#[test]
fn c2_edge_correction_removes_the_bias() {
    // CSR: K(r) = πr²; without a correction K is too low near a quarter of the side
    let mut rng = Rng::new(5);
    let (mut none, mut trans) = (0.0, 0.0);
    for _ in 0..20 {
        let pts = csr(100, 100.0, 100.0, &mut rng);
        none += k_hat(&pts, 100.0, 100.0, 25.0, false) / 20.0;
        trans += k_hat(&pts, 100.0, 100.0, 25.0, true) / 20.0;
    }
    let theory = PI * 25.0 * 25.0;
    assert!(none < 0.9 * theory, "no correction, mean K(25) = {none:.0}: expected clearly below πr² = {theory:.0}");
    assert!(close(trans, theory, 0.06 * theory), "translation, mean K(25) = {trans:.0}: expected about πr² = {theory:.0}");
}

#[test]
fn c3_pair_correlation_reads_the_pattern() {
    let mut rng = Rng::new(11);
    let random = csr(400, 200.0, 200.0, &mut rng);
    let g = pair_correlation(&random, 200.0, 200.0, 15.0, 3.0);
    assert!(close(g, 1.0, 0.15), "CSR: g(15) = {g:.3}, expected about 1");
    let regular = hard_core(200, 8.0, 200.0, 200.0, &mut rng);
    let g = pair_correlation(&regular, 200.0, 200.0, 4.0, 2.0);
    assert_eq!(g, 0.0, "hard core 8 µm: g(4) must be 0 (no pairs closer than 8)");
    let clustered = thomas(0.001, 6.0, 3.0, 200.0, 200.0, &mut rng);
    let g = pair_correlation(&clustered, 200.0, 200.0, 4.0, 1.5);
    assert!(g > 3.0, "clusters with σ = 3 µm: g(4) = {g:.2}, expected far above 1");
}
