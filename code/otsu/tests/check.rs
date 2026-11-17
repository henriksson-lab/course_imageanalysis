// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p otsu

use otsu::*;

/// The worked example of the lecture: grey levels 0–7, 20 pixels.
const EXAMPLE: [usize; 8] = [2, 5, 3, 1, 1, 2, 4, 2];
/// B3: three groups of grey levels.
const THREE: [usize; 8] = [9, 0, 0, 8, 7, 0, 0, 5];

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// Expected counts of a mixture of two normal classes over 256 grey levels, `n` pixels.
fn mixture(n: f64, w1: f64, (m0, s0): (f64, f64), (m1, s1): (f64, f64)) -> Vec<usize> {
    let pdf = |x: f64, m: f64, s: f64| (-(x - m).powi(2) / (2.0 * s * s)).exp() / (s * (2.0 * std::f64::consts::PI).sqrt());
    (0..256).map(|i| (n * ((1.0 - w1) * pdf(i as f64, m0, s0) + w1 * pdf(i as f64, m1, s1))).round() as usize).collect()
}

#[test]
fn r1_class_stats_b1() {
    for (t, w0, mu0, mu1) in [(2, 0.50, 1.10, 5.50), (3, 0.55, 14.0 / 11.0, 52.0 / 9.0), (4, 0.60, 1.50, 6.00)] {
        let (a, b, c) = class_stats(&EXAMPLE, t);
        assert!(close(a, w0, 1e-9) && close(b, mu0, 1e-9) && close(c, mu1, 1e-9), "B1 (a), t = {t}: got ({a:.3}, {b:.3}, {c:.3}), expected ({w0:.2}, {mu0:.2}, {mu1:.2})");
    }
}

#[test]
fn r2_otsu_b1() {
    assert_eq!(otsu(&EXAMPLE), 3, "B1 (b): Otsu's threshold of the worked example");
}

#[test]
fn r2_otsu_b3() {
    // {0} | {3, 4, 7} (4.05) beats {0, 3, 4} | {7} (3.33); t = 0, 1, 2 tie, the smallest wins
    assert_eq!(otsu(&THREE), 0, "B3 (b): Otsu's threshold of the three groups");
}

#[test]
fn r3_isodata_b2() {
    let t = isodata(&EXAMPLE, 5.5);
    assert!(close(t, 3.53, 0.005), "B2: isodata from 5.5: got {t:.3}, expected 3.53");
}

#[test]
fn r3_isodata_b3() {
    let a = isodata(&THREE, 6.5);
    let b = isodata(&THREE, 0.5);
    assert!(close(a, 4.58, 0.005), "B3 (a): isodata from 6.5: got {a:.3}, expected 4.58");
    assert!(close(b, 2.17, 0.005), "B3 (c): isodata from 0.5: got {b:.3}, expected 2.17");
}

#[test]
fn r4_min_error_near_bayes_b4() {
    // B4: background 50, objects 150, both sigma 20, 10 % objects. Bayes: 108.8.
    // Otsu errs into the large background class (98.5 on these curves).
    let h = mixture(1e6, 0.1, (50.0, 20.0), (150.0, 20.0));
    let t = min_error(&h);
    let o = otsu(&h);
    assert!((106..=111).contains(&t), "minimum error: got t = {t}, expected about 108 (Bayes 108.8)");
    assert!((96..=100).contains(&o), "Otsu: got t = {o}, expected about 98 (B4 (c))");
}

#[test]
fn r4_min_error_equal_classes() {
    // equal weights and widths: the crossing is the midpoint, 100
    let h = mixture(1e6, 0.5, (50.0, 20.0), (150.0, 20.0));
    let t = min_error(&h);
    assert!((99..=100).contains(&t), "minimum error, equal classes: got t = {t}, expected 99 or 100");
}

#[test]
fn r5_otsu3_three_groups() {
    // B3's three groups {0}, {3, 4}, {7}: the first split anywhere in 0–2, the second in 4–6
    assert_eq!(otsu3(&THREE), (0, 4), "three classes of B3");
}

#[test]
fn r5_otsu3_three_peaks() {
    let a = mixture(1e6, 0.5, (40.0, 8.0), (120.0, 8.0));
    let b = mixture(5e5, 0.0, (200.0, 8.0), (0.0, 1.0));
    let h: Vec<usize> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    let (t1, t2) = otsu3(&h);
    assert!((75..=85).contains(&t1) && (155..=165).contains(&t2), "peaks at 40, 120, 200: got ({t1}, {t2}), expected about (80, 160)");
}
