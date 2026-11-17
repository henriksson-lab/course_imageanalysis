// Checks for the coding exercises of "Domain shift and adaptation", using the numbers of the
// pen-and-paper exercises. Run: cargo test -p finetuning domain_shift::

use finetuning::domain_shift::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn cell(i: f32, s: f32) -> Cell {
    Cell { f: [i, s], class: 0 }
}

#[test]
fn c1_stats() {
    let (m, s) = stats(&[cell(0.0, 0.0), cell(2.0, 4.0)]);
    assert!(close(m[0], 1.0, 1e-6) && close(m[1], 2.0, 1e-6), "means of (0, 0) and (2, 4): got {m:?}, expected [1, 2]");
    assert!(close(s[0], 1.0, 1e-6) && close(s[1], 2.0, 1e-6), "SDs (divided by n): got {s:?}, expected [1, 2]");
}

#[test]
fn c2_standardise() {
    let z = standardise(&[cell(0.6, 0.5), Cell { f: [0.3, 0.7], class: 1 }], [0.4, 0.5], [0.1, 0.2]);
    assert!(close(z[0].f[0], 2.0, 1e-5) && close(z[0].f[1], 0.0, 1e-5), "(0.6, 0.5): got {:?}, expected [2, 0]", z[0].f);
    assert!(close(z[1].f[0], -1.0, 1e-5) && close(z[1].f[1], 1.0, 1e-5), "(0.3, 0.7): got {:?}, expected [-1, 1]", z[1].f);
    assert_eq!(z[1].class, 1, "the class is kept");
}

#[test]
fn c3_shift_score() {
    let a = [cell(0.4, 0.5), cell(0.6, 0.7)]; // means 0.5, 0.6; SDs 0.1, 0.1
    let b = [cell(0.2, 0.5), cell(0.4, 0.7)]; // intensity mean 0.3: 2 SDs lower
    assert!(close(shift_score(&a, &b), 2.0, 1e-4), "intensity 2 SDs lower: got {}", shift_score(&a, &b));
    assert!(close(shift_score(&a, &a), 0.0, 1e-6), "no shift: score 0");
}

#[test]
fn c4_augment() {
    let cells = [Cell { f: [0.5, 0.3], class: 1 }, cell(0.4, 0.6)];
    let aug = augment(&cells, 4, 7);
    assert_eq!(aug.len(), 8, "4 copies of 2 cells");
    for (k, c) in aug.iter().enumerate() {
        let orig = cells[k / 4];
        assert!(close(c.f[1], orig.f[1], 1e-7) && c.class == orig.class, "copy {k}: size and class must stay");
        let lo = 0.4 * orig.f[0] - 0.05;
        let hi = 1.2 * orig.f[0] + 0.2;
        assert!(c.f[0] >= lo - 1e-6 && c.f[0] <= hi + 1e-6, "copy {k}: intensity {} outside [{lo}, {hi}]", c.f[0]);
    }
}
