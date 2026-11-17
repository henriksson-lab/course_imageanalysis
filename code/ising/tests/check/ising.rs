// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p ising ising::

use imgcourse::Image;
use ising::ising::*;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn c1_energy_2x2_b1() {
    let aligned = Image::from_rows(&[[1i8, 1], [1, 1]]);
    let checker = Image::from_rows(&[[1i8, -1], [-1, 1]]);
    let one_flipped = Image::from_rows(&[[-1i8, 1], [1, 1]]);
    let two_side_by_side = Image::from_rows(&[[-1i8, 1], [-1, 1]]);
    for (name, s, expected) in [("all +1", &aligned, -4.0), ("checkerboard", &checker, 4.0), ("one −1", &one_flipped, 0.0), ("two −1 side by side", &two_side_by_side, 0.0)] {
        let e = energy(s, 1.0, 0.0, false);
        assert!(close(e as f64, expected, 1e-5), "B1, {name}: got E = {e}, expected {expected}");
    }
}

#[test]
fn c1_energy_with_field_b2_b5() {
    let plus = Image::new(2, 2, 1i8);
    let minus = Image::new(2, 2, -1i8);
    let (ep, em) = (energy(&plus, 1.0, 0.5, false), energy(&minus, 1.0, 0.5, false));
    assert!(close(ep as f64, -6.0, 1e-5) && close(em as f64, -2.0, 1e-5), "B2(a): got {ep} and {em}, expected −6 and −2");
    // B5: a row of 4 pixels, J = 1, h = 0.5
    let a = Image::from_rows(&[[1i8, 1, -1, -1]]);
    let b = Image::from_rows(&[[1i8, -1, 1, -1]]);
    let (ea, eb) = (energy(&a, 1.0, 0.5, false), energy(&b, 1.0, 0.5, false));
    assert!(close(ea as f64, -1.0, 1e-5), "B5(a), (+, +, −, −): got {ea}, expected −1");
    assert!(close(eb as f64, 3.0, 1e-5), "B5(a), (+, −, +, −): got {eb}, expected 3");
}

#[test]
fn c1_energy_periodic() {
    // the 3 × 3 torus has 18 pairs; a 3 × 1 ring has 3
    let e = energy(&Image::new(3, 3, 1i8), 1.0, 0.0, true);
    assert!(close(e as f64, -18.0, 1e-5), "3 × 3 periodic, all +1: got {e}, expected −18 (18 pairs)");
    let e = energy(&Image::new(3, 1, 1i8), 1.0, 0.0, true);
    assert!(close(e as f64, -3.0, 1e-5), "ring of 3, all +1: got {e}, expected −3");
    let e = energy(&Image::new(2, 2, 1i8), 1.0, 0.0, true);
    assert!(close(e as f64, -4.0, 1e-5), "2 × 2 periodic: no extra pairs on sides of 2: got {e}, expected −4");
}

#[test]
fn c2_levels_2x2_b1() {
    let levels = energy_levels(2, 2, 1.0, 0.0, false);
    let expected = [(-4.0, 2), (0.0, 12), (4.0, 2)];
    assert_eq!(levels.len(), 3, "B1(a): got the levels {levels:?}, expected {expected:?}");
    for ((e, n), (ee, en)) in levels.iter().zip(expected) {
        assert!(close(*e as f64, ee, 1e-4) && *n == en, "B1(a): got the levels {levels:?}, expected {expected:?}");
    }
}

#[test]
fn c2_levels_3x3_periodic_e2() {
    let levels = energy_levels(3, 3, 1.0, 0.0, true);
    let expected = [(-18.0, 2), (-10.0, 18), (-6.0, 48), (-2.0, 198), (2.0, 144), (6.0, 102)];
    assert_eq!(levels.len(), 6, "E2: got the levels {levels:?}, expected {expected:?}");
    for ((e, n), (ee, en)) in levels.iter().zip(expected) {
        assert!(close(*e as f64, ee, 1e-4) && *n == en, "E2: got the levels {levels:?}, expected {expected:?}");
    }
}

#[test]
fn c3_partition_function_b1() {
    let levels = [(-4.0, 2), (0.0, 12), (4.0, 2)];
    let (z, p) = level_probabilities(&levels, 1.0);
    assert!(close(z, 121.23, 0.01), "B1(b), T = 1: got Z = {z:.3}, expected 121.23");
    assert!(close(p[0], 0.901, 0.001), "B1(c), T = 1: got p(E = −4) = {:.4}, expected 0.901", p[0]);
    let (z, p) = level_probabilities(&levels, 4.0);
    assert!(close(z, 18.17, 0.01), "B1(b), T = 4: got Z = {z:.3}, expected 18.17");
    let expected = [0.299, 0.660, 0.040];
    for k in 0..3 {
        assert!(close(p[k], expected[k], 0.001), "B1(c, d), T = 4: got {p:.3?}, expected {expected:?}");
    }
    let sum: f64 = p.iter().sum();
    assert!(close(sum, 1.0, 1e-9), "the probabilities should sum to 1, got {sum}");
}

#[test]
fn c3_field_and_torus_b2_e2() {
    let (z, _) = level_probabilities(&energy_levels(2, 2, 1.0, 0.5, false), 1.0);
    assert!(close(z, 427.20, 0.01), "B2(c), h = 0.5: got Z = {z:.3}, expected 427.20");
    let (z, p) = level_probabilities(&energy_levels(3, 3, 1.0, 0.0, true), 2.0);
    assert!(close(z, 20438.0, 1.0), "E2, 3 × 3 torus, T = 2: got Z = {z:.1}, expected 20 438");
    assert!(close(p[0], 0.793, 0.001), "E2: got p(ground states) = {:.4}, expected 0.793", p[0]);
}

#[test]
fn c4_transfer_matrix_b9() {
    let z = ring_partition_function(3, 1.0, 0.0, 1.0);
    assert!(close(z, 42.38, 0.01), "B9, ring of 3: got Z = {z:.3}, expected 42.38");
}

#[test]
fn c4_transfer_matrix_equals_listing() {
    // a ring of 10 spins with a field, against listing all 1024 states
    for (j, h, t) in [(1.0, 0.0, 1.0), (1.0, 0.3, 1.5), (-0.5, 0.2, 2.0)] {
        let (z_list, _) = level_probabilities(&energy_levels(10, 1, j as f32, h as f32, true), t);
        let z = ring_partition_function(10, j, h, t);
        assert!((z - z_list).abs() < 1e-3 * z_list, "ring of 10, J = {j}, h = {h}, T = {t}: transfer matrix {z:.4}, listing {z_list:.4}");
    }
}
