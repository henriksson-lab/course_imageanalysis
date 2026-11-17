// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day46 flim_fret::

use day46::flim_fret::*;
use std::f32::consts::PI;

/// 80 MHz: ω = 2π · 0.08 rad/ns.
const OMEGA: f32 = 2.0 * PI * 0.08;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn single(tau: f32) -> (f32, f32) {
    let wt = OMEGA * tau;
    (1.0 / (1.0 + wt * wt), wt / (1.0 + wt * wt))
}

#[test]
fn c1_efficiency_and_distance_b1() {
    let e = fret_efficiency(1.5, 2.5);
    assert!(close(e, 0.4, 1e-5), "B1: τD = 2.5, τDA = 1.5: got E = {e}, expected 0.4");
    for (e, r) in [(0.4, 5.35), (0.9, 3.47), (0.1, 7.21), (0.5, 5.0)] {
        let got = distance(e, 5.0);
        assert!(close(got, r, 0.01), "B1: E = {e}, R0 = 5 nm: got r = {got:.3} nm, expected {r}");
    }
}

#[test]
fn c2_photon_fraction_b2_b8() {
    let f = photon_fraction(0.3, 1.0, 2.5);
    assert!(close(f, 0.146, 0.001), "B2(a): α = 0.3, τDA = 1.0, τD = 2.5: got f = {f:.4}, expected 0.146");
    let a = interacting_fraction(f, 1.0, 2.5);
    assert!(close(a, 0.3, 1e-4), "B8: going back from f = {f:.4} must give α = 0.3, got {a}");
    let f = photon_fraction(5.0 / 7.0, 1.0, 2.5);
    assert!(close(f, 0.5, 1e-4), "B3: 5/7 of donors at E = 0.6 give half the photons, got f = {f}");
    for alpha in [0.1, 0.5, 0.9] {
        assert!(photon_fraction(alpha, 1.2, 2.5) < alpha, "B8: f must be smaller than α = {alpha} when E > 0");
    }
}

#[test]
fn c3_line_through_the_donor_b2() {
    // B2: donor 2.5 ns, 30 % of donors at E = 0.6 (τDA = 1.0 ns): the pixel at 0.854 P_D + 0.146 P_DA.
    let (pd, pda) = (single(2.5), single(1.0));
    let f = 0.1463;
    let p = ((1.0 - f) * pd.0 + f * pda.0, (1.0 - f) * pd.1 + f * pda.1);
    let (tau_da, got_f) = fret_from_line(pd, p, OMEGA);
    assert!(close(tau_da, 1.0, 0.005), "B2: the line through the pixel ({:.3}, {:.3}) must meet the semicircle at τDA = 1.0 ns, got {tau_da:.3}", p.0, p.1);
    assert!(close(got_f, f, 0.001), "B2: photon fraction of the FRET state: got {got_f:.4}, expected {f}");
    let e = fret_efficiency(tau_da, 2.5);
    assert!(close(e, 0.6, 0.003), "B2: E from the line: got {e:.3}, expected 0.6");
}

#[test]
fn c4_barycentric_b5_b9() {
    let (free, bound, label) = ((0.961, 0.193), (0.388, 0.487), (0.163, 0.370));
    let f = barycentric((0.63, 0.32), free, bound, label);
    let expected = [0.495, 0.320, 0.185];
    for k in 0..3 {
        assert!(close(f[k], expected[k], 0.003), "B9: got fractions {f:?}, expected {expected:?}");
    }
    assert!(close(f.iter().sum::<f32>(), 1.0, 1e-5), "the fractions must sum to 1");
    // A point outside the triangle has a negative fraction.
    let out = barycentric((0.9, 0.45), free, bound, label);
    assert!(out.iter().any(|&v| v < 0.0), "the point (0.9, 0.45) is outside the triangle, but all fractions are ≥ 0: {out:?}");
    // The corners themselves.
    let corner = barycentric(bound, free, bound, label);
    assert!(close(corner[1], 1.0, 1e-5) && close(corner[0], 0.0, 1e-5), "at the bound corner: got {corner:?}, expected [0, 1, 0]");
}
