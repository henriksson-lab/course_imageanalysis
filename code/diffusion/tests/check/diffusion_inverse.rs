// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p diffusion diffusion_inverse::

use diffusion::diffusion_inverse::*;
use imgcourse::Rng;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_seen_plus_unseen() {
    let x: Vec<f32> = (0..32).map(|i| ((i * 7) % 5) as f32 - 0.3 * i as f32).collect();
    let (seen, unseen) = split_measured(&x, 6);
    for i in 0..32 {
        assert!(close(seen[i] + unseen[i], x[i], 1e-4), "position {i}: seen + unseen = {}, expected x = {}", seen[i] + unseen[i], x[i]);
    }
    let m = measure(&unseen, 6);
    assert!(m.iter().all(|v| v.abs() < 1e-3), "the measurement of the unseen part: got {m:?}, expected all 0 (B8)");
    let (_, unseen_all) = split_measured(&x, 32);
    assert!(unseen_all.iter().all(|v| v.abs() < 1e-3), "keeping all 32 cosines, nothing should be unseen");
    let (seen_c, _) = split_measured(&[2.0; 16], 1);
    assert!(seen_c.iter().all(|&v| close(v, 2.0, 1e-4)), "a constant signal is seen by the first cosine alone");
}

/// B1: three training images (1, 0), (0, 1), (0.5, 0.5); the camera measures their mean.
fn b1() -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {
    let signals = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
    let measured = signals.iter().map(|s| vec![(s[0] + s[1]) / 2.0]).collect();
    (signals, measured, vec![0.4, 0.4, 0.2])
}

#[test]
fn c2_weights_b1() {
    let (_, measured, prior) = b1();
    let w = posterior_weights(&[0.5], &measured, &prior, 0.05);
    for j in 0..3 {
        assert!(close(w[j], prior[j], 1e-5), "B1a: got weights {w:?}, expected the prior weights 0.4, 0.4, 0.2");
    }
    // a measurement that one candidate explains much better
    let w = posterior_weights(&[1.0, 0.0], &[vec![1.0, 0.0], vec![0.0, 1.0]], &[0.5, 0.5], 0.5);
    let want = 1.0 / (1.0 + (-4.0f32).exp());
    assert!(close(w[0], want, 1e-4), "two candidates at distance² 0 and 2, σ = 0.5: got {}, expected {want:.4}", w[0]);
    let w = posterior_weights(&[100.0], &[vec![0.0], vec![1.0]], &[0.5, 0.5], 0.1);
    assert!(w.iter().all(|v| v.is_finite()) && close(w[1], 1.0, 1e-4), "far from both candidates: got {w:?}, expected [0, 1] (no NaN)");
}

#[test]
fn c3_mean_b1() {
    let (signals, _, prior) = b1();
    let m = posterior_mean(&prior, &signals);
    assert!(close(m[0], 0.5, 1e-5) && close(m[1], 0.5, 1e-5), "B1b: got {m:?}, expected (0.5, 0.5)");
}

#[test]
fn c4_samples_b1() {
    let mut rng = Rng::new(1);
    let n = 20_000;
    let pairs = (0..n).filter(|_| posterior_sample(&[0.4, 0.4, 0.2], &mut rng) < 2).count() as f32 / n as f32;
    assert!(close(pairs, 0.8, 0.015), "B1c: a resolved pair in {pairs:.3} of the samples, expected 0.8");
}

#[test]
fn c5_coverage() {
    let samples: Vec<Vec<f32>> = (0..100).map(|j| vec![j as f32 / 100.0; 4]).collect(); // values 0.00 ... 0.99
    let c = band_coverage(&[0.5, 0.01, 0.97, 0.2], &samples);
    assert!(close(c, 0.5, 1e-6), "truth 0.5, 0.01, 0.97, 0.2 against a band from 0.05 to 0.95: got {c}, expected 2 of 4 = 0.5");
}
