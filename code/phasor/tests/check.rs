// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p phasor

use phasor::*;
use std::f32::consts::PI;

/// 80 MHz: ω = 2π · 0.08 rad/ns.
const OMEGA: f32 = 2.0 * PI * 0.08;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_four_bins_b7() {
    let (g, s) = phasor(&[8.0, 4.0, 2.0, 1.0], 12.5);
    assert!(close(g, 0.141, 0.001) && close(s, 0.424, 0.001), "B7(a): got ({g:.3}, {s:.3}), expected (0.141, 0.424)");
}

#[test]
fn c1_fine_histogram_lands_on_the_semicircle() {
    // A 2.5 ns decay in 1000 bins over 12.5 ns (the periodic decay has the same shape).
    let n = 1000;
    let counts: Vec<f32> = (0..n).map(|j| (-((j as f32 + 0.5) * 12.5 / n as f32) / 2.5).exp()).collect();
    let (g, s) = phasor(&counts, 12.5);
    assert!(close(g, 0.388, 0.002) && close(s, 0.487, 0.002), "2.5 ns: got ({g:.3}, {s:.3}), expected (0.388, 0.487)");
    // Doubling every count changes nothing: the phasor is normalised.
    let double: Vec<f32> = counts.iter().map(|c| 2.0 * c).collect();
    let (g2, s2) = phasor(&double, 12.5);
    assert!(close(g, g2, 1e-5) && close(s, s2, 1e-5), "twice the photons must give the same phasor");
}

#[test]
fn c1_mixtures_add_by_photon_fraction() {
    let n = 500;
    let decay = |tau: f32, photons: f32| -> Vec<f32> {
        let raw: Vec<f32> = (0..n).map(|j| (-((j as f32 + 0.5) * 12.5 / n as f32) / tau).exp()).collect();
        let s: f32 = raw.iter().sum();
        raw.iter().map(|v| photons * v / s).collect()
    };
    let (a, b) = (decay(0.4, 300.0), decay(4.0, 700.0));
    let mix: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    let (pa, pb, pm) = (phasor(&a, 12.5), phasor(&b, 12.5), phasor(&mix, 12.5));
    let expected = (0.3 * pa.0 + 0.7 * pb.0, 0.3 * pa.1 + 0.7 * pb.1);
    assert!(close(pm.0, expected.0, 1e-4) && close(pm.1, expected.1, 1e-4), "300 + 700 photons: got {pm:?}, expected 0.3 P_a + 0.7 P_b = {expected:?}");
}

#[test]
fn c2_single_phasor_b1_b3() {
    let (g, s) = single_phasor(2.0, OMEGA);
    assert!(close(g, 0.497, 0.001) && close(s, 0.500, 0.001), "B1: τ = 2 ns: got ({g:.3}, {s:.3}), expected (0.497, 0.500)");
    let (g, s) = single_phasor(0.4, OMEGA);
    assert!(close(g, 0.961, 0.001) && close(s, 0.193, 0.001), "B3: τ = 0.4 ns: got ({g:.3}, {s:.3}), expected (0.961, 0.193)");
    for tau in [0.1, 1.0, 5.0, 20.0] {
        let (g, s) = single_phasor(tau, OMEGA);
        let r = ((g - 0.5).powi(2) + s * s).sqrt();
        assert!(close(r, 0.5, 1e-5), "τ = {tau}: the point ({g}, {s}) must lie on the semicircle (distance {r} from (0.5, 0))");
    }
}

#[test]
fn c2_phase_and_modulation_b2_b5() {
    let (tp, tm) = phase_and_modulation_lifetimes(0.8, 0.4, OMEGA);
    assert!(close(tp, 0.99, 0.01) && close(tm, 0.99, 0.01), "B2, (0.8, 0.4): got τφ = {tp:.3}, τm = {tm:.3}, expected 0.99 and 0.99");
    let (tp, tm) = phase_and_modulation_lifetimes(0.6, 0.3, OMEGA);
    assert!(close(tp, 0.99, 0.01) && close(tm, 2.20, 0.01), "B2, (0.6, 0.3): got τφ = {tp:.3}, τm = {tm:.3}, expected 0.99 and 2.20");
    let (tp, tm) = phase_and_modulation_lifetimes(0.310, 0.390, OMEGA);
    assert!(close(tp, 2.50, 0.01) && close(tm, 3.46, 0.02), "B5, 20 % background: got τφ = {tp:.3}, τm = {tm:.3}, expected 2.50 and 3.46");
}

#[test]
fn c3_calibration_b6() {
    let polar = |phi_deg: f32, m: f32| (m * phi_deg.to_radians().cos(), m * phi_deg.to_radians().sin());
    let (g, s) = calibrate(polar(50.0, 0.60), polar(70.0, 0.40), 4.0, OMEGA);
    let (phi, m) = (s.atan2(g).to_degrees(), (g * g + s * s).sqrt());
    assert!(close(phi, 43.6, 0.1) && close(m, 0.668, 0.002), "B6(a): got φ = {phi:.1}°, m = {m:.3}, expected 43.6° and 0.668");
    // The reference itself must land on its true point.
    let r = calibrate(polar(70.0, 0.40), polar(70.0, 0.40), 4.0, OMEGA);
    let t = single_phasor(4.0, OMEGA);
    assert!(close(r.0, t.0, 1e-5) && close(r.1, t.1, 1e-5), "the reference must be corrected to its true phasor {t:?}, got {r:?}");
}

#[test]
fn c4_lever_rule_b4() {
    let (short, long) = ((0.917, 0.276), (0.244, 0.430));
    let f = fraction_along((0.715, 0.322), short, long);
    assert!(close(f, 0.30, 0.005), "B4: got {f:.3} of the way from 0.6 ns to 3.5 ns, expected 0.30");
    assert!(close(fraction_along(short, short, long), 0.0, 1e-6) && close(fraction_along(long, short, long), 1.0, 1e-6), "the end points must give 0 and 1");
}
