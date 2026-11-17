// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p diffusion diffusion_sampling::

use diffusion::diffusion_sampling::*;
use imgcourse::Rng;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// The best denoiser for Gaussian data with sd 1: it multiplies by 1 / (1 + σ²) (B3).
fn gauss(y: f32, sigma: f32) -> f32 {
    y / (1.0 + sigma * sigma)
}

#[test]
fn c1_steps_b3() {
    let start = 10f32.sqrt(); // one standard deviation of the points at σ = 3
    let cases: [(&[f32], f32, &str); 3] = [(&[3.0, 0.0], 0.316, "B3a"), (&[3.0, 1.5, 0.0], 0.535, "B3b"), (&[3.0, 1.0, 0.0], 0.632, "B3d")];
    for (sigmas, want, name) in cases {
        let got = euler_sample(start, sigmas, gauss);
        assert!(close(got, want, 0.002), "{name}, levels {sigmas:?}: a point at one sd ends at {got:.3}, expected {want} (the spread)");
    }
    let ten: Vec<f32> = (0..=10).map(|k| 3.0 - 0.3 * k as f32).collect();
    let got = euler_sample(start, &ten, gauss);
    assert!(close(got, 0.89, 0.01), "B3c, ten equal steps: got {got:.3}, expected about 0.89");
}

#[test]
fn c2_frechet_b4() {
    // a set of two values μ ± s has mean μ and sd s
    let data = [-1.0, 1.0];
    for (s, want) in [(0.316, 0.468), (0.535, 0.216), (1.0, 0.0)] {
        let d = frechet_1d(&data, &[-s, s]);
        assert!(close(d, want, 0.001), "N(0, {s}²) against N(0, 1): got {d:.3}, expected {want} (B4)");
    }
    let d = frechet_1d(&[0.0, 2.0], &[3.0, 5.0]);
    assert!(close(d, 9.0, 1e-5), "same sd, means 1 and 4: got {d}, expected 9");
}

#[test]
fn c2_frechet_misses_the_gap_b5() {
    // two modes at ±2 (sd 0.5) against one wide Gaussian with the same mean and variance
    let mut rng = Rng::new(1);
    let modes: Vec<f32> = (0..20_000).map(|i| if i % 2 == 0 { 2.0 } else { -2.0 } + 0.5 * rng.normal()).collect();
    let wide: Vec<f32> = (0..20_000).map(|_| 4.25f32.sqrt() * rng.normal()).collect();
    let d = frechet_1d(&modes, &wide);
    assert!(d < 0.01, "B5b: got Fréchet distance {d:.4}, expected about 0");
}

#[test]
fn c3_precision_recall() {
    let real: Vec<f32> = (0..200).map(|i| if i % 2 == 0 { -2.0 } else { 2.0 } + 0.01 * (i / 2) as f32).collect();
    let (p, r) = precision_recall(&real, &real, 3);
    assert!(close(p, 1.0, 1e-6) && close(r, 1.0, 1e-6), "the real points against themselves: got ({p}, {r}), expected (1, 1)");
    // mode dropping: generate only the right mode
    let right: Vec<f32> = real.iter().copied().filter(|&x| x > 0.0).collect();
    let (p, r) = precision_recall(&real, &right, 3);
    assert!(close(p, 1.0, 1e-6) && close(r, 0.5, 1e-6), "only one of two modes: got ({p}, {r}), expected precision 1, recall 0.5");
    // everything between the modes
    let middle = vec![0.0; 50];
    let (p, r) = precision_recall(&real, &middle, 3);
    assert!(close(p, 0.0, 1e-6) && close(r, 0.0, 1e-6), "all samples between the modes: got ({p}, {r}), expected (0, 0)");
}

#[test]
fn c4_copies_b6() {
    let train = [0.1, 0.5, 2.0];
    let d = nearest_distances(&[0.5, 1.9, -1.0], &train);
    let want = [0.0, 0.1, 1.1];
    for i in 0..3 {
        assert!(close(d[i], want[i], 1e-5), "nearest training distances: got {d:?}, expected {want:?}");
    }
}
