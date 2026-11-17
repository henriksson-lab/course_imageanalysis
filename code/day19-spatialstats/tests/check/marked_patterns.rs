// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day19 marked_patterns::

use imgcourse::Rng;
use day19::marked_patterns::*;
use std::f64::consts::PI;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn l_minus_r(k: f64, r: f64) -> f64 {
    (k / PI).sqrt() - r
}

#[test]
fn c1_cross_k_b1() {
    // 5 tumour and 4 immune cells in 10 × 10 µm, placed so that the close pairs are those of B1:
    // I1–T1 1.41, I2–T2 1.41, I3–T3 2.00, I4–T4 2.83; all other pairs are more than 3 µm apart
    let tumour = [(1.0, 1.0), (8.0, 1.0), (1.0, 8.0), (8.0, 8.0), (4.5, 9.5)];
    let immune = [(2.0, 2.0), (9.0, 2.0), (1.0, 6.0), (6.0, 6.0)];
    for (r, k_expected, l_expected) in [(1.0, 0.0, -1.0), (2.0, 15.0, 0.185), (3.0, 20.0, -0.477)] {
        let k = cross_k(&immune, &tumour, 10.0, 10.0, r, false);
        assert!(close(k, k_expected, 1e-9), "B1(b), r = {r}: K10 = {k}, expected {k_expected}");
        assert!(close(l_minus_r(k, r), l_expected, 2e-3), "B1(b), r = {r}: L10 - r = {:.3}, expected {l_expected}", l_minus_r(k, r));
        let k01 = cross_k(&tumour, &immune, 10.0, 10.0, r, false);
        assert!(close(k01, k, 1e-9), "r = {r}: K01 = {k01} but K10 = {k}; without weights they must be equal");
    }
}

#[test]
fn c1_independent_types() {
    let mut rng = Rng::new(21);
    let (t, i) = (thomas(0.001, 6.0, 4.0, 200.0, 200.0, &mut rng), csr(150, 200.0, 200.0, &mut rng));
    let k = cross_k(&t, &i, 200.0, 200.0, 15.0, true);
    assert!(close(k, PI * 225.0, 0.2 * PI * 225.0), "clustered tumour cells, independent random immune cells: K01(15) = {k:.0}, expected about πr² = {:.0}", PI * 225.0);
}

#[test]
fn c2_permute_labels() {
    let labels: Vec<u8> = (0..20).map(|k| (k % 3 == 0) as u8).collect();
    let mut rng = Rng::new(1);
    let p = permute_labels(&labels, &mut rng);
    let ones = |v: &[u8]| v.iter().filter(|&&l| l == 1).count();
    assert_eq!(p.len(), 20, "the permuted labels must have the same length");
    assert_eq!(ones(&p), ones(&labels), "permuting must keep the number of each label");
    assert_ne!(p, labels, "after a shuffle of 20 labels the order should change");
    // every position equally likely: the single 1 of [1, 0, 0, 0] lands at each place about 25 % of the time
    let mut hits = [0; 4];
    for _ in 0..4000 {
        let q = permute_labels(&[1, 0, 0, 0], &mut rng);
        hits[q.iter().position(|&l| l == 1).unwrap()] += 1;
    }
    for (k, &h) in hits.iter().enumerate() {
        assert!((800..1200).contains(&h), "position {k}: the 1 landed there {h} times of 4000, expected about 1000");
    }
}

#[test]
fn c3_mark_connection_b3() {
    let mut rng = Rng::new(4);
    let pts = csr(600, 200.0, 200.0, &mut rng);
    // random labels, 30 % type 1: p01 = 2 · 0.3 · 0.7 = 0.42 at every r
    let labels: Vec<u8> = (0..600).map(|_| (rng.uniform() < 0.3) as u8).collect();
    let p = mark_connection(&pts, &labels, 10.0, 2.0);
    let p1 = labels.iter().filter(|&&l| l == 1).count() as f64 / 600.0;
    let expected = 2.0 * p1 * (1.0 - p1);
    assert!(close(p, expected, 0.05), "B3(a), random labels: p01(10) = {p:.3}, expected about 2 p0 p1 = {expected:.3}");
    // segregated: type 0 on the left half, type 1 on the right
    let halves: Vec<u8> = pts.iter().map(|p| (p.0 > 100.0) as u8).collect();
    let p = mark_connection(&pts, &halves, 5.0, 2.0);
    assert!(p < 0.1, "segregated halves: p01(5) = {p:.3}, expected far below 0.42");
}

#[test]
fn c4_inhomogeneous_k() {
    // a gradient (no interaction): CSR thinned with probability x / 200
    let mut rng = Rng::new(8);
    let pts: Vec<Point> = csr(600, 200.0, 200.0, &mut rng).into_iter().filter(|p| (rng.uniform() as f64) < p.0 / 200.0).collect();
    let r = 20.0;
    let n = pts.len() as f64;
    let mut plain = 0.0;
    for (i, &p) in pts.iter().enumerate() {
        for (j, &q) in pts.iter().enumerate() {
            if i != j && dist(p, q) <= r {
                plain += translation_weight(q.0 - p.0, q.1 - p.1, 200.0, 200.0);
            }
        }
    }
    let plain = l_minus_r(40000.0 / (n * (n - 1.0)) * plain, r);
    let inhom = l_minus_r(k_inhom(&pts, 200.0, 200.0, r, 40.0), r);
    assert!(plain > 1.5, "the plain L(20) - 20 of a gradient should look clustered: got {plain:.2}");
    assert!(inhom.abs() < 1.5, "the inhomogeneous L(20) - 20 (σ = 40) should be about 0: got {inhom:.2}");
}
