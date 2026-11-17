// Checks for the coding exercises, using the numbers of the lecture and the pen-and-paper
// exercises. Run: cargo test -p gmm gaussian_mixture::

use gmm::gaussian_mixture::*;
use imgcourse::Rng;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// The lecture's example, Σ = [[5, 2], [2, 2]] centred at 0.
const EXAMPLE: Gaussian = Gaussian { mu: [0.0, 0.0], sigma: [[5.0, 2.0], [2.0, 2.0]] };

#[test]
fn c1_mahalanobis_b2() {
    let p = mahalanobis([2.0, 1.0], &EXAMPLE);
    let q = mahalanobis([-1.0, 2.0], &EXAMPLE);
    assert!(close(p, (5.0f32 / 6.0).sqrt(), 1e-4), "B2: d_M(P) = {p}, expected √(5/6) = 0.913");
    assert!(close(q, 5.0f32.sqrt(), 1e-4), "B2: d_M(Q) = {q}, expected √5 = 2.236");
    // In 1D terms: with Σ = σ² I it is the Euclidean distance over σ.
    let round = Gaussian { mu: [1.0, 1.0], sigma: [[4.0, 0.0], [0.0, 4.0]] };
    let d = mahalanobis([4.0, 5.0], &round);
    assert!(close(d, 2.5, 1e-5), "Σ = 4 I: d_M = {d}, expected 5 / 2 = 2.5");
}

#[test]
fn c2_density_lecture() {
    let peak = density([0.0, 0.0], &EXAMPLE);
    let expected = 1.0 / (2.0 * std::f32::consts::PI * 6.0f32.sqrt());
    assert!(close(peak, expected, 1e-5), "peak height: got {peak}, expected 1/(2π√6) = {expected} (B11)");
    let p = density([2.0, 1.0], &EXAMPLE);
    let q = density([-1.0, 2.0], &EXAMPLE);
    assert!(close(p, 0.043, 1e-3), "density at P: got {p}, expected 0.043");
    assert!(close(q, 0.0053, 3e-4), "density at Q: got {q}, expected 0.005");
}

#[test]
fn c3_responsibilities_b4() {
    // B4 in 2D: σ = 20 along x1, and every point has x2 = 0, so x2 plays no role.
    let g = |m: f32| Gaussian { mu: [m, 0.0], sigma: [[400.0, 0.0], [0.0, 1.0]] };
    let comps = [g(100.0), g(160.0)];
    for (x, e) in [(120.0, 0.91), (130.0, 0.70), (140.0, 0.34)] {
        let r = responsibilities([x, 0.0], &[0.7, 0.3], &comps);
        assert!(close(r[0], e, 0.01), "B4(a): r1({x}) = {}, expected {e}", r[0]);
        assert!(close(r[0] + r[1], 1.0, 1e-5), "the responsibilities should add up to 1, got {r:?}");
    }
    let r = responsibilities([135.6, 0.0], &[0.7, 0.3], &comps);
    assert!(close(r[0], 0.5, 0.01), "B4(b): the boundary is at 135.6, but r1(135.6) = {}", r[0]);
    let r = responsibilities([130.0, 0.0], &[0.5, 0.5], &comps);
    assert!(close(r[0], 0.5, 1e-4), "B4(c): with equal weights r1(130) = {}, expected 0.5", r[0]);
}

#[test]
fn c4_fit_gaussian_b5() {
    let g = fit_gaussian(&[[0.0, 0.0], [2.0, 2.0], [1.0, 3.0], [3.0, 1.0]]);
    assert!(close(g.mu[0], 1.5, 1e-5) && close(g.mu[1], 1.5, 1e-5), "B5: mean {:?}, expected (1.5, 1.5)", g.mu);
    let expected = [[1.25, 0.25], [0.25, 1.25]];
    for i in 0..2 {
        for j in 0..2 {
            assert!(close(g.sigma[i][j], expected[i][j], 1e-5), "B5: covariance {:?}, expected {expected:?}", g.sigma);
        }
    }
}

#[test]
fn c5_sample_mixture() {
    let comps = [EXAMPLE, Gaussian { mu: [8.0, -3.0], sigma: [[1.0, -0.5], [-0.5, 2.0]] }];
    let mut rng = Rng::new(1);
    let (points, labels) = sample_mixture(&[0.3, 0.7], &comps, 20000, &mut rng);
    let share = labels.iter().filter(|&&l| l == 0).count() as f32 / 20000.0;
    assert!(close(share, 0.3, 0.015), "share of component 1: got {share}, expected 0.3");
    for k in 0..2 {
        let mine: Vec<[f32; 2]> = (0..points.len()).filter(|&i| labels[i] == k).map(|i| points[i]).collect();
        let g = fit_gaussian(&mine);
        for i in 0..2 {
            assert!(close(g.mu[i], comps[k].mu[i], 0.1), "component {}: sample mean {:?}, expected {:?}", k + 1, g.mu, comps[k].mu);
            for j in 0..2 {
                assert!(close(g.sigma[i][j], comps[k].sigma[i][j], 0.25), "component {}: sample covariance {:?}, expected {:?}", k + 1, g.sigma, comps[k].sigma);
            }
        }
    }
}
