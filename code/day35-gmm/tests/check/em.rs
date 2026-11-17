// Checks for the coding exercises, using the numbers of the lecture and the pen-and-paper
// exercises. Run: cargo test -p day35 em::

use day35::em::*;
use imgcourse::Rng;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

const B1_DATA: [f32; 6] = [1.0, 2.0, 4.0, 7.0, 9.0, 10.0];

fn b1_start() -> Mixture {
    Mixture { weights: vec![0.5, 0.5], means: vec![2.0, 8.0], variances: vec![4.0, 4.0] }
}

#[test]
fn c1_e_step_b1() {
    let (r, ll) = e_step(&B1_DATA, &b1_start());
    let expected = [0.998, 0.989, 0.818, 0.047, 0.002, 0.001];
    for i in 0..6 {
        assert!(close(r[i][0], expected[i], 2e-3), "B1(a): r_1 of the six points: got {:?}, expected {expected:?}", r.iter().map(|ri| ri[0]).collect::<Vec<_>>());
        assert!(close(r[i][0] + r[i][1], 1.0, 1e-5), "the responsibilities of point {} should add up to 1", B1_DATA[i]);
    }
    assert!(close(ll, -14.94, 0.01), "B1(d): log-likelihood of the start: got {ll}, expected −14.94");
}

#[test]
fn c1_e_step_b2() {
    let start = Mixture { weights: vec![0.5, 0.5], means: vec![0.0, 4.0], variances: vec![1.0, 1.0] };
    let (r, _) = e_step(&[0.0, 2.0, 3.0], &start);
    let expected = [0.9997, 0.5, 0.018];
    for i in 0..3 {
        assert!(close(r[i][0], expected[i], 1e-3), "B2(a): r_1 = {:?}, expected {expected:?}", r.iter().map(|ri| ri[0]).collect::<Vec<_>>());
    }
}

#[test]
fn c2_m_step_b1() {
    let (r, _) = e_step(&B1_DATA, &b1_start());
    let m = m_step(&B1_DATA, &r, 0.0);
    assert!(close(m.weights[0], 0.475, 0.005), "B1(b): π_1 = {}, expected 0.48 (N_1 = 2.85 of 6)", m.weights[0]);
    assert!(close(m.means[0], 2.31, 0.01) && close(m.means[1], 8.39, 0.01), "B1(b): μ = {:?}, expected (2.31, 8.39)", m.means);
    assert!(close(m.variances[0], 1.87, 0.01) && close(m.variances[1], 2.83, 0.01), "B1(b): σ² = {:?}, expected (1.87, 2.83)", m.variances);
}

#[test]
fn c2_m_step_b2() {
    let start = Mixture { weights: vec![0.5, 0.5], means: vec![0.0, 4.0], variances: vec![1.0, 1.0] };
    let data = [0.0, 2.0, 3.0];
    let (r, _) = e_step(&data, &start);
    let m = m_step(&data, &r, 0.0);
    assert!(close(m.means[0], 0.69, 0.01) && close(m.means[1], 2.66, 0.01), "B2(b): μ = {:?}, expected (0.69, 2.66)", m.means);
    assert!(close(m.variances[0], 0.94, 0.01) && close(m.variances[1], 0.23, 0.01), "B2(b): σ² = {:?}, expected (0.94, 0.23)", m.variances);
    assert!(close(m.weights[0], 0.51, 0.01), "B2(b): π_1 = {}, expected 0.51", m.weights[0]);
    // The variance floor.
    let m = m_step(&data, &r, 0.5);
    assert!(close(m.variances[1], 0.5, 1e-6), "with floor 0.5, σ_2² should be 0.5, got {}", m.variances[1]);
}

#[test]
fn c3_em_b1() {
    let (m, trace) = em(&B1_DATA, &b1_start(), 0.0, 1e-5, 1000);
    let first = [-14.94, -14.22, -14.03, -13.99];
    for k in 0..4 {
        assert!(close(trace[k], first[k], 0.01), "B1(d): log-likelihood trace starts {:?}, expected {first:?}", &trace[..4.min(trace.len())]);
    }
    for w in trace.windows(2) {
        assert!(w[1] >= w[0] - 1e-4, "the log-likelihood must never go down: {trace:?}");
    }
    assert!(close(m.means[0], 2.34, 0.01) && close(m.means[1], 8.66, 0.01), "B1(c): converged μ = {:?}, expected (2.34, 8.66)", m.means);
    assert!(close(m.variances[0], 1.57, 0.02) && close(m.variances[1], 1.57, 0.02), "converged σ² = {:?}, expected 1.57 for both", m.variances);
}

#[test]
fn c3_em_never_goes_down() {
    let mut rng = Rng::new(4);
    let data: Vec<f32> = (0..300).map(|i| if i % 3 == 0 { 5.0 + rng.normal() } else { 2.0 * rng.normal() }).collect();
    let start = Mixture { weights: vec![0.3, 0.3, 0.4], means: vec![-1.0, 0.0, 1.0], variances: vec![1.0, 1.0, 1.0] };
    let (_, trace) = em(&data, &start, 1e-3, 1e-6, 500);
    for w in trace.windows(2) {
        assert!(w[1] >= w[0] - 1e-2, "the log-likelihood went down from {} to {}", w[0], w[1]);
    }
}

#[test]
fn c4_bic_aic_b4() {
    let ll = [-4210.0, -4020.0, -4012.0, -4010.0];
    let bic = [8433.8, 8074.5, 8079.3, 8096.0];
    let aic = [8424.0, 8050.0, 8040.0, 8042.0];
    for k in 0..4 {
        let (b, a) = bic_aic(ll[k], parameters(k + 1), 1000);
        assert!(close(b, bic[k], 0.1), "B4, K = {}: BIC = {b}, expected {}", k + 1, bic[k]);
        assert!(close(a, aic[k], 0.01), "B4, K = {}: AIC = {a}, expected {}", k + 1, aic[k]);
    }
    // B5: diagonal (p = 9) and full (p = 11) covariances, n = 200.
    let (diag, _) = bic_aic(273.0, 9, 200);
    let (full, _) = bic_aic(274.7, 11, 200);
    assert!(close(diag, -498.3, 0.1) && close(full, -491.1, 0.1), "B5: BIC {diag} and {full}, expected −498.3 and −491.1");
}
