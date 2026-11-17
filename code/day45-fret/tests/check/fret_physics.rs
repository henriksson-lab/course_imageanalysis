// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day45 fret_physics::

use day45::fret_physics::sim;
use day45::fret_physics::*;
use imgcourse::Rng;

fn close(got: f32, expected: f32, tol: f32) -> bool {
    (got - expected).abs() < tol
}

#[test]
fn c1_efficiency_b1() {
    let expected = [(0.5, 0.985), (0.8, 0.792), (1.0, 0.5), (1.2, 0.251), (1.5, 0.081), (2.0, 0.015)];
    for (ratio, e) in expected {
        let got = efficiency(ratio * 5.0, 5.0);
        assert!(close(got, e, 1e-3), "B1, r/R0 = {ratio}: got E = {got:.3}, expected {e}");
    }
}

#[test]
fn c1_apparent_distance_b2() {
    for (e, r) in [(0.30, 5.76), (0.35, 5.54), (0.25, 6.00), (0.90, 3.47)] {
        let got = apparent_distance(e, 5.0);
        assert!(close(got, r, 0.01), "B2, E = {e}: got r = {got:.2} nm, expected {r}");
    }
    let back = efficiency(apparent_distance(0.42, 4.95), 4.95);
    assert!(close(back, 0.42, 1e-5), "apparent_distance should undo efficiency: got {back}");
}

#[test]
fn c2_forster_radius_of_the_pair() {
    let l = sim::wavelengths();
    let donor: Vec<f32> = l.iter().map(|&x| sim::donor_emission(x)).collect();
    let acc: Vec<f32> = l.iter().map(|&x| sim::acceptor_extinction(x, 0.0)).collect();
    let j = overlap_integral(&l, &donor, &acc);
    assert!((j / 1e15 - 2.41).abs() < 0.02, "J of the lecture's pair: got {:.3e}, expected 2.41e15", j);
    let r0 = forster_radius(sim::Q_DONOR, 2.0 / 3.0, sim::N_MEDIUM, j);
    assert!(close(r0, 4.95, 0.01), "R0 of the lecture's pair: got {r0:.3} nm, expected 4.95");
    // B5: twice the extinction coefficient: R0 × 2^(1/6)
    let acc2: Vec<f32> = acc.iter().map(|e| 2.0 * e).collect();
    let r2 = forster_radius(sim::Q_DONOR, 2.0 / 3.0, sim::N_MEDIUM, overlap_integral(&l, &donor, &acc2));
    assert!(close(r2, 5.56, 0.01), "B5, acceptor C: got R0 = {r2:.2} nm, expected 5.56");
    // the donor emission does not need to be normalised by the caller
    let donor3: Vec<f32> = donor.iter().map(|f| 3.0 * f).collect();
    let j3 = overlap_integral(&l, &donor3, &acc);
    assert!((j3 / j - 1.0).abs() < 1e-6, "scaling the donor emission must not change J (normalise it to unit area)");
}

#[test]
fn c2_forster_radius_b3() {
    // B3: from R0 = 5 nm, κ² 4 instead of 2/3 gives 6.74 nm, κ² 0.1 gives 3.64 nm
    let j = 1e15;
    let r = forster_radius(0.4, 2.0 / 3.0, 1.4, j);
    let r4 = forster_radius(0.4, 4.0, 1.4, j);
    let r01 = forster_radius(0.4, 0.1, 1.4, j);
    assert!(close(5.0 * r4 / r, 6.74, 0.01) && close(5.0 * r01 / r, 3.64, 0.01), "B3(c, d): got {:.2} and {:.2}", 5.0 * r4 / r, 5.0 * r01 / r);
}

#[test]
fn c3_kappa2_b9() {
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let x = [1.0, 0.0, 0.0];
    let cases = [
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 4.0, "(a) collinear"),
        ([0.0, 1.0, 0.0], [0.0, 1.0, 0.0], 1.0, "(b) parallel, side by side"),
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0, "(c) perpendicular"),
        ([s, s, 0.0], [s, s, 0.0], 0.25, "(d) both at 45°"),
    ];
    for (d, a, e, what) in cases {
        let k = kappa2(d, a, x);
        assert!(close(k, e, 1e-5), "B9{what}: got κ² = {k}, expected {e}");
    }
}

#[test]
fn c4_mean_kappa2_b10() {
    let mut rng = Rng::new(1);
    let k = mean_kappa2(20000, &mut rng);
    assert!(close(k, 2.0 / 3.0, 0.02), "B10: the mean of κ² should be close to 2/3, got {k}");
}

#[test]
fn c4_mixture_b4() {
    let mut rng = Rng::new(1);
    let e = mean_efficiency(&[2.5, 5.0], 5.0, false, &mut rng);
    assert!(close(e, 0.742, 1e-3), "B4(b): got mean E = {e}, expected 0.742");
    let r = apparent_distance(e, 5.0) / 5.0;
    assert!(close(r, 0.84, 0.005), "B4(c): got apparent distance {r:.3} R0, expected 0.84 R0");
    // frozen random orientations lower the mean efficiency at r = R0
    let distances = vec![5.0; 4000];
    let e = mean_efficiency(&distances, 5.0, true, &mut rng);
    assert!(e < 0.45 && e > 0.3, "frozen orientations at r = R0: got mean E = {e}, expected well below 0.5");
}
