// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p spatialstats envelopes::

use spatialstats::envelopes::*;
use imgcourse::Rng;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// The 19 simulated values of L(5) - 5 of B1.
const B1: [f64; 19] = [-0.9, 0.3, -0.2, 1.1, 0.6, -0.5, 0.0, 0.8, -1.3, 0.4, -0.1, 0.9, 1.4, -0.7, 0.2, 0.5, -0.4, 1.0, 0.1];

/// The ten simulated curves of B3 and the observed one.
fn b3() -> (Vec<Vec<f64>>, Vec<f64>) {
    let sims = [
        [0.4, -0.3, 0.6, 0.9, -0.8],
        [-0.5, 0.2, -0.4, 0.3, 1.1],
        [0.1, 0.7, 0.2, -0.6, -0.4],
        [0.8, 0.1, -0.9, 0.5, 0.6],
        [-0.2, -0.6, 0.3, 1.2, 0.2],
        [0.3, 0.4, 0.1, -0.3, -1.0],
        [-0.7, 0.2, 0.5, 0.4, 0.3],
        [0.2, -0.9, 0.8, -0.2, 0.7],
        [0.6, 0.3, -0.2, 0.1, -0.5],
        [-0.1, 0.5, 0.4, -0.8, 0.9],
    ];
    (sims.iter().map(|c| c.to_vec()).collect(), vec![0.5, 1.0, 1.3, 0.9, 0.6])
}

#[test]
fn c1_monte_carlo_p_b1() {
    let p = mc_p_value(1.2, &B1, true);
    assert!(close(p, 0.10, 1e-12), "B1(a), observed 1.2, clustering side: p = {p}, expected 2/20 = 0.10");
    let p = mc_p_value(1.2, &B1, false);
    assert!(close(p, 0.95, 1e-12), "B1(a), observed 1.2, regular side: p = {p}, expected 19/20 = 0.95");
    let p = mc_p_value(1.6, &B1, true);
    assert!(close(p, 0.05, 1e-12), "B1(b), observed 1.6: p = {p}, expected 1/20 = 0.05");
}

#[test]
fn c2_envelope_and_escapes_b3() {
    let (sims, obs) = b3();
    let (lower, upper) = pointwise_envelope(&sims);
    let (lo, hi) = ([-0.7, -0.9, -0.9, -0.8, -1.0], [0.8, 0.7, 0.8, 1.2, 1.1]);
    for k in 0..5 {
        assert!(close(lower[k], lo[k], 1e-12) && close(upper[k], hi[k], 1e-12), "B3(a), radius r{}: envelope [{}, {}], expected [{}, {}]", k + 1, lower[k], upper[k], lo[k], hi[k]);
    }
    let e = escapes(&obs, &lower, &upper);
    assert_eq!(e, vec![1, 2], "B3(a): the observed curve leaves the band at r2 and r3 (indices 1 and 2)");
}

#[test]
fn c3_mad_test_b3() {
    let (sims, obs) = b3();
    let (t, p) = mad_test(&obs, &sims);
    assert!(close(t, 1.3, 1e-12), "B3(b): T of the observed curve = {t}, expected 1.3");
    assert!(close(p, 1.0 / 11.0, 1e-12), "B3(b): p = {p}, expected 1/11 = 0.091");
}

#[test]
fn c4_false_alarms() {
    // 60 random patterns of 40 points, 19 simulations, 10 radii: the pointwise band raises far
    // more false alarms than the MAD test (whose rate is 5 % on average)
    let mut rng = Rng::new(9);
    let radii: Vec<f64> = (1..=10).map(|k| 2.0 * k as f64).collect();
    let (pointwise, mad) = false_alarm_rates(40, 100.0, 100.0, &radii, 19, 60, &mut rng);
    assert!(pointwise > 0.2, "pointwise envelope: {:.0} % of random patterns rejected, expected far more than 10 %", 100.0 * pointwise);
    assert!(mad <= 0.15, "MAD test: {:.0} % of random patterns rejected, expected about 5 %", 100.0 * mad);
}
