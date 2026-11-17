// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day39 diffusion::

use day39::diffusion::*;
use imgcourse::Rng;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// The signal levels for `n` steps from pure noise (τ = 1) to the data (τ = 0).
fn levels(n: usize) -> Vec<f32> {
    (0..=n).map(|k| cosine_alpha_bar(1.0 - k as f32 / n as f32)).collect()
}

#[test]
fn c1_schedule_lecture() {
    // The lecture's figure: signal √ᾱ = 1.00, 0.95, 0.80, 0.58, 0.01 at τ = 0, 0.2, 0.4, 0.6, 1.
    for (tau, signal) in [(0.0, 1.0), (0.2, 0.95), (0.4, 0.80), (0.6, 0.58), (1.0, 0.0)] {
        let got = cosine_alpha_bar(tau).sqrt();
        assert!(close(got, signal, 0.015), "signal at τ = {tau}: got {got}, expected {signal}");
    }
}

#[test]
fn c1_forward_b1() {
    let got = forward(&[2.0], 0.64, &[-1.0]);
    assert!(close(got[0], 1.0, 1e-5), "B1 (a): 0.8 · 2 + 0.6 · (−1): got {}, expected 1.0", got[0]);
}

#[test]
fn c2_best_denoiser_b1_b2() {
    let pm1 = vec![vec![1.0], vec![-1.0]];
    let got = best_x0(&[0.5], 0.36, &pm1, 0.0)[0];
    assert!(close(got, 0.437, 1e-3), "B2: data ±1, √ᾱ = 0.6, x = 0.5: got {got}, expected tanh(0.469) = 0.437");
    let got = best_x0(&[1.0], 0.64, &[vec![2.0]], 0.0)[0];
    assert!(close(got, 2.0, 1e-5), "B1 (b): with one training value the best guess is that value, got {got}");
    let got = best_x0(&[3.0], 0.0, &pm1, 0.0)[0];
    assert!(close(got, 0.0, 1e-5), "at pure noise the best guess is the mean of the data, 0, got {got}");
    // One blob: the posterior mean of a Gaussian (B10).
    let got = best_x0(&[1.0], 0.5, &[vec![0.0]], 1.0)[0];
    assert!(close(got, 0.5f32.sqrt(), 1e-4), "one blob N(0, 1), ᾱ = 0.5, x = 1: got {got}, expected √0.5");
    let far = best_x0(&[100.0], 0.36, &pm1, 0.0)[0];
    assert!(far.is_finite() && close(far, 1.0, 1e-4), "far from the data the answer should still be finite (1), got {far}");
}

#[test]
fn c3_ddim_few_and_many_steps_b3() {
    let blobs = vec![vec![-1.2], vec![1.2]];
    let mut rng = Rng::new(1);
    let mut far_one = 0;
    let mut on_blob = 0;
    for _ in 0..200 {
        let start = [rng.normal()];
        let one = sample_ddim(&start, &levels(1), &blobs, 0.1)[0];
        if one.abs() < 0.05 {
            far_one += 1;
        }
        let many = sample_ddim(&start, &levels(100), &blobs, 0.1)[0];
        if (many.abs() - 1.2).abs() < 0.4 {
            on_blob += 1;
        }
    }
    assert_eq!(far_one, 200, "B3 (a): one step from pure noise puts every sample at the mean 0");
    assert_eq!(on_blob, 200, "B3 (b): 100 steps should put every sample in a blob, got {on_blob} of 200");
}

#[test]
fn c4_ddpm_lands_in_the_blobs_in_proportion() {
    let blobs = vec![vec![-1.2, 0.0], vec![1.2, 0.0]];
    let mut rng = Rng::new(2);
    let (mut left, mut off) = (0, 0);
    for _ in 0..400 {
        let start = [rng.normal(), rng.normal()];
        let x = sample_ddpm(&start, &levels(100), &blobs, 0.1, &mut rng);
        let d = ((x[0].abs() - 1.2).powi(2) + x[1].powi(2)).sqrt();
        if d > 0.4 {
            off += 1;
        }
        if x[0] < 0.0 {
            left += 1;
        }
    }
    assert_eq!(off, 0, "100 stochastic steps: no sample should land off the blobs, got {off} of 400");
    assert!((150..=250).contains(&left), "the two blobs are equally likely: got {left} of 400 on the left");
}

#[test]
fn c5_posterior_b9() {
    let (m, v) = posterior(0.0, 1.0, 0.5, 1.0);
    assert!(close(m, 0.8, 1e-5) && close(v, 0.2, 1e-5), "B9: prior N(0, 1), noise variance 0.25, y = 1: got mean {m}, variance {v}, expected 0.8, 0.2");
    // A sample from the posterior has twice the squared error of the mean.
    let mut rng = Rng::new(5);
    let (mut err_mean, mut err_sample) = (0.0, 0.0);
    let n = 20000;
    for _ in 0..n {
        let x = rng.normal();
        let y = x + 0.5 * rng.normal();
        let (m, v) = posterior(0.0, 1.0, 0.5, y);
        let sample = m + v.sqrt() * rng.normal();
        err_mean += (m - x).powi(2) / n as f32;
        err_sample += (sample - x).powi(2) / n as f32;
    }
    assert!(close(err_mean, 0.2, 0.02) && close(err_sample, 0.4, 0.03), "B9: errors of mean and sample: got {err_mean}, {err_sample}, expected 0.2 and 0.4");
}
