// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p viola-jones

use viola_jones::*;

/// Exercise B1: one feature with values 1..6 and labels −, −, +, −, +, +.
fn b1() -> (Vec<f32>, Vec<f64>) {
    (vec![1., 2., 3., 4., 5., 6.], vec![-1., -1., 1., -1., 1., 1.])
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_best_stump_b1a() {
    let (v, y) = b1();
    let (t, p, e) = best_stump(&v, &y, &[1. / 6.; 6]);
    assert!(close(t as f64, 2.5, 1e-6) && p == 1.0, "B1(a): got threshold {t}, polarity {p}; expected \"face above 2.5\"");
    assert!(close(e, 1. / 6., 1e-9), "B1(a): weighted error {e}, expected 1/6");
}

#[test]
fn c1_best_stump_round_2_b1c() {
    let (v, y) = b1();
    let (t, p, e) = best_stump(&v, &y, &[0.1, 0.1, 0.1, 0.5, 0.1, 0.1]);
    assert!(close(t as f64, 4.5, 1e-6) && p == 1.0, "B1(c): got threshold {t}, polarity {p}; expected \"face above 4.5\"");
    assert!(close(e, 0.1, 1e-9), "B1(c): weighted error {e}, expected 0.1");
}

#[test]
fn c1_best_stump_other_direction() {
    // Faces have small values here: the best stump says "face if below".
    let (t, p, e) = best_stump(&[5., 1., 4., 2.], &[-1., 1., -1., 1.], &[0.25; 4]);
    assert!(p == -1.0 && e.abs() < 1e-9 && t > 2.0 && t < 4.0, "got threshold {t}, polarity {p}, error {e}; expected \"face below\" a threshold between 2 and 4, error 0");
}

#[test]
fn c2_adaboost_b1() {
    let (v, y) = b1();
    let stumps = adaboost(&[v], &y, 3);
    let expected = [(2.5, 1.0, 1. / 6., 0.5 * 5f64.ln()), (4.5, 1.0, 0.1, 0.5 * 9f64.ln()), (3.5, -1.0, 2. / 9., 0.5 * 3.5f64.ln())];
    for (r, (s, (t, p, e, a))) in stumps.iter().zip(expected).enumerate() {
        assert!(close(s.threshold as f64, t, 1e-6) && s.polarity == p, "round {}: got threshold {}, polarity {}; expected {t}, {p}", r + 1, s.threshold, s.polarity);
        assert!(close(s.error, e, 1e-6), "round {}: error {}, expected {e:.4}", r + 1, s.error);
        assert!(close(s.alpha, a, 1e-6), "round {}: alpha {}, expected {a:.4}", r + 1, s.alpha);
    }
}

#[test]
fn c3_scores_b1() {
    let (v, y) = b1();
    let stumps = adaboost(std::slice::from_ref(&v), &y, 3);
    // B1(c): after two rounds 3 is still wrong; B1(d): after three rounds none is.
    let two: Vec<f64> = v.iter().map(|&x| score(&stumps[..2], |_| x)).collect();
    let expected = [-1.903, -1.903, -0.294, -0.294, 1.903, 1.903];
    for i in 0..6 {
        assert!(close(two[i], expected[i], 1e-3), "B1(c), value {}: score {:.3}, expected {}", i + 1, two[i], expected[i]);
    }
    let three: Vec<f64> = v.iter().map(|&x| score(&stumps, |_| x)).collect();
    let expected = [-1.277, -1.277, 0.332, -0.920, 1.277, 1.277];
    for i in 0..6 {
        assert!(close(three[i], expected[i], 1e-3), "B1(d), value {}: score {:.3}, expected {}", i + 1, three[i], expected[i]);
    }
}

#[test]
fn c4_cascade_b2_b3() {
    let b2 = [Stage { d: 0.99, f: 0.5, n: 2 }, Stage { d: 0.98, f: 0.3, n: 5 }, Stage { d: 0.97, f: 0.2, n: 20 }];
    let (d, f, n) = cascade_rates(&b2);
    assert!(close(d, 0.941, 1e-3), "B2(a): D = {d:.4}, expected 0.941");
    assert!(close(f, 0.03, 1e-9), "B2(a): F = {f}, expected 0.03");
    assert!(close(n, 7.5, 1e-9), "B2(b): {n} features per background window, expected 7.5");
    let b3 = [Stage { d: 0.99, f: 0.3, n: 1 }; 10];
    let (d, f, _) = cascade_rates(&b3);
    assert!(close(d, 0.904, 1e-3) && close(f, 5.9e-6, 1e-7), "B3(a): D = {d:.4}, F = {f:.2e}; expected 0.904 and 5.9e-6");
}
