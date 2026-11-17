// Checks for the coding exercises of "Training GANs and what goes wrong", with the numbers of the
// pen-and-paper exercises. Run: cargo test -p gans gan_training::

use gans::common::normal_pdf;
use gans::gan_training::*;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_js_divergence_b2() {
    let p = [0.5, 0.5, 0.0, 0.0];
    let q = [0.0, 0.0, 0.5, 0.5];
    let js = js_divergence(&p, &q, 1.0);
    assert!(close(js, 2f64.ln(), 1e-12), "B2: no overlap gives ln 2 = 0.693; got {js}");
    let js = js_divergence(&p, &p, 1.0);
    assert!(close(js, 0.0, 1e-12), "identical densities give 0; got {js}");
    let js = js_divergence(&[0.5, 0.5, 0.0], &[0.0, 0.5, 0.5], 1.0);
    assert!(close(js, 0.3466, 1e-4), "B2: half the mass shared gives ln 2 / 2 = 0.347; got {js}");
    // two far-apart Gaussians on a grid
    let dx = 0.01;
    let grid: Vec<f64> = (0..3000).map(|i| -10.0 + i as f64 * dx).collect();
    let p: Vec<f64> = grid.iter().map(|&x| normal_pdf(x, 0.0, 0.5)).collect();
    let q: Vec<f64> = grid.iter().map(|&x| normal_pdf(x, 8.0, 0.5)).collect();
    assert!(close(js_divergence(&p, &q, dx), 2f64.ln(), 1e-6), "far apart: ln 2");
}

#[test]
fn c2_wasserstein_b2() {
    let w = wasserstein1(&[0.0, 1.0, 2.0], &[3.0, 1.0, 2.0]);
    assert!(close(w, 1.0, 1e-12), "B2: (0, 1, 2) against (1, 2, 3) gives 1; got {w}");
    let w = wasserstein1(&[0.0, 0.0], &[5.0, 5.0]);
    assert!(close(w, 5.0, 1e-12), "two point masses 5 apart: 5; got {w}");
}

#[test]
fn c3_precision_recall_b5() {
    let r = knn_radii(&[0.0, 1.0, 3.0], 1);
    assert_eq!(r, vec![1.0, 1.0, 2.0], "k = 1 radii of (0, 1, 3)");
    let (p, rc) = precision_recall(&[0.0, 1.0, 2.0, 3.0], &[0.1, 1.1, 10.0, 11.0], 1);
    assert!(close(p, 0.5, 1e-12) && close(rc, 0.75, 1e-12), "B5: precision 0.5, recall 0.75; got {p}, {rc}");
}
