// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p vae

use vae::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

// The lecture's example: x = 1.5, w = 1, sigma_x = 0.5.
const X: f32 = 1.5;
const W: f32 = 1.0;
const SX: f32 = 0.5;

#[test]
fn c1_reparameterise_b1() {
    for (eps, z) in [(-1.0, 0.5), (0.0, 1.0), (0.5, 1.25)] {
        let got = reparameterise(1.0, 0.5, eps);
        assert!(close(got, z, 1e-6), "B1: mu = 1, s = 0.5, eps = {eps}: got z = {got}, expected {z}");
    }
}

#[test]
fn c1_kl_b4_b5() {
    for (mu, s, kl) in [(0.5, 1.0, 0.125), (0.0, 2.0, 0.807), (2.0, 0.1, 3.808), (0.0, 1.0, 0.0)] {
        let got = kl_to_prior(&[mu], &[s]);
        assert!(close(got, kl, 1e-3), "B4: KL of (mu, s) = ({mu}, {s}): got {got}, expected {kl}");
    }
    let got = kl_to_prior(&[1.0, 0.0], &[0.5, 1.0]);
    assert!(close(got, 0.818, 1e-3), "B5: 2D cloud, centre (1, 0), spreads (0.5, 1): got {got}, expected 0.818");
}

#[test]
fn c2_reconstruction_lecture_table_and_b6() {
    for (mu, s, r) in [(0.0, 1.0, 6.5), (1.5, 0.05, 0.005), (1.2, 0.2f32.sqrt(), 0.58), (1.0, 0.5, 1.0)] {
        let got = reconstruction_exact(X, W, SX, mu, s);
        assert!(close(got, r, 1e-3), "reconstruction term at mu = {mu}, s = {s}: got {got}, expected {r}");
    }
}

#[test]
fn c2_draws_estimate_the_exact_term() {
    // eps = −1 and +1 have mean 0 and mean square 1: the estimate is then exact.
    let got = reconstruction_from_draws(X, W, SX, 0.0, 1.0, &[-1.0, 1.0]);
    assert!(close(got, 6.5, 1e-4), "draws eps = −1, +1 at the prior: got {got}, expected exactly 6.5");
    // Many random draws: close to the exact value.
    let mut rng = imgcourse::Rng::new(1);
    let eps: Vec<f32> = (0..20000).map(|_| rng.normal()).collect();
    let got = reconstruction_from_draws(X, W, SX, 1.2, 0.447, &eps);
    assert!(close(got, 0.58, 0.02), "20000 draws at the best cloud: got {got}, expected about 0.58");
}

fn average_end(path: &[(f32, f32)]) -> (f32, f32) {
    let tail = &path[path.len() - 500..];
    let n = tail.len() as f32;
    (tail.iter().map(|p| p.0).sum::<f32>() / n, tail.iter().map(|p| p.1).sum::<f32>() / n)
}

#[test]
fn c3_fit_cloud_finds_the_best_cloud_b10() {
    let path = fit_cloud(X, W, SX, 1.0, 2000, 10, 0.05, 1);
    assert_eq!(path.len(), 2001, "the path has the start and one entry per step");
    assert!(path[0] == (0.0, 1.0), "the path starts at the prior (0, 1), got {:?}", path[0]);
    let (mu, s) = average_end(&path);
    assert!(close(mu, 1.2, 0.05) && close(s, 0.447, 0.05), "beta = 1: got mu = {mu}, s = {s}, expected 1.2, 0.447");
    let (mu, s) = average_end(&fit_cloud(X, W, SX, 5.0, 2000, 10, 0.05, 2));
    assert!(close(mu, 0.667, 0.05) && close(s, 0.745, 0.05), "beta = 5: got mu = {mu}, s = {s}, expected 0.667, 0.745");
}
