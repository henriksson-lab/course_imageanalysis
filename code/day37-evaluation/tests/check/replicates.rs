// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day37 replicates::

use day37::replicates::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// The experiment means of B1: control and treated, days 1 to 3.
const CONTROL: [f32; 3] = [11.0, 14.0, 8.0];
const TREATED: [f32; 3] = [13.0, 15.0, 10.0];

#[test]
fn c1_mean_sd_b1() {
    let (m, s) = mean_sd(&CONTROL);
    assert!(close(m, 11.0, 1e-5) && close(s, 3.0, 1e-4), "B1c, control means: got mean {m}, SD {s}, expected 11 and 3.0");
    let (m, s) = mean_sd(&[2.0, 1.0, 2.0]);
    assert!(close(m, 5.0 / 3.0, 1e-5) && close(s, 0.577, 1e-3), "B1b, differences 2, 1, 2: got mean {m}, SD {s}, expected 1.67 and 0.58");
}

#[test]
fn c2_paired_b1b() {
    let t = paired_t(&CONTROL, &TREATED);
    assert!(close(t.estimate, 1.667, 1e-3) && close(t.se, 0.333, 1e-3), "B1b: got estimate {}, SE {}, expected 1.67 and 0.33", t.estimate, t.se);
    assert!(close(t.t, 5.0, 1e-3) && t.df == 2, "B1b: got t = {} with {} df, expected 5.0 with 2 df", t.t, t.df);
    assert!(t.significant(), "B1b: t = 5.0 > 4.30 is significant");
    let (lo, hi) = t.interval();
    assert!(close(lo, 0.23, 0.01) && close(hi, 3.10, 0.01), "B1b, 95 % CI: got {lo} to {hi}, expected 0.23 to 3.10");
}

#[test]
fn c2_unpaired_b1c() {
    let t = unpaired_t(&CONTROL, &TREATED);
    assert!(close(t.se, 2.26, 0.01) && close(t.t, 0.74, 0.01) && t.df == 4, "B1c: got SE {}, t = {}, {} df, expected 2.26, 0.74, 4 df", t.se, t.t, t.df);
    assert!(!t.significant(), "B1c: t = 0.74 < 2.78 is not significant");
}

#[test]
fn c2_cells_as_n_b1d() {
    let control = [10., 11., 12., 11., 14., 15., 13., 14., 8., 9., 7., 8.];
    let treated = [12., 13., 14., 13., 15., 16., 14., 15., 10., 9., 11., 10.];
    let t = unpaired_t(&control, &treated);
    assert!(close(t.t, 1.65, 0.02) && t.df == 22, "B1d, cells as n: got t = {} with {} df, expected about 1.65 with 22 df", t.t, t.df);
}

#[test]
fn c3_false_positives_of_cells_as_n() {
    // Lecture: 3 experiments per condition, 100 cells, day SD 1, cell SD 2, no effect:
    // cells as n gives p < 0.05 in about 69 % of the studies, experiment means in about 5 %.
    let (cells, means) = false_positive_rates(3, 100, 1.0, 2.0, 2000, 1);
    assert!((0.63..0.75).contains(&cells), "cells as n: got a false-positive rate of {cells:.3}, expected about 0.69");
    assert!((0.02..0.08).contains(&means), "experiment means: got a false-positive rate of {means:.3}, expected about 0.05");
    // Without a day effect, cells really are independent (demo question P6).
    let (cells, _) = false_positive_rates(3, 100, 0.0, 2.0, 2000, 2);
    assert!((0.03..0.08).contains(&cells), "no day effect: cells as n should be about 0.05, got {cells:.3}");
}

const B3: [f32; 10] = [0.001, 0.004, 0.012, 0.02, 0.03, 0.2, 0.4, 0.5, 0.7, 0.9];

#[test]
fn c4_bonferroni_b3() {
    assert_eq!(bonferroni(&B3, 0.05), 2, "B3b: Bonferroni");
}

#[test]
fn c4_benjamini_hochberg_b3() {
    assert_eq!(benjamini_hochberg(&B3, 0.05), 4, "B3c: Benjamini–Hochberg");
    let mut shuffled = B3;
    shuffled.reverse();
    assert_eq!(benjamini_hochberg(&shuffled, 0.05), 4, "B3c with the p-values in another order");
    // the largest k counts even if a smaller k fails: 0.03 > 0.01 at k = 1, but 0.03 <= 0.03 at k = 3
    assert_eq!(benjamini_hochberg(&[0.03, 0.03, 0.03, 0.9, 0.9], 0.05), 3, "three p-values of 0.03 among 5");
}
