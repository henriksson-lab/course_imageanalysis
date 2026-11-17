// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p guidance

use guidance::*;
use imgcourse::Rng;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_guided_b1() {
    for (w, want) in [(0.0, 0.5), (1.0, 0.8), (3.0, 1.4), (-1.0, 0.2)] {
        let g = guided(0.5, 0.2, w);
        assert!(close(g, want, 1e-5), "w = {w}: got {g}, expected {want} (B1)");
    }
}

#[test]
fn c2_posterior_b2() {
    // B2: A = N(+1, 1), B = N(−1, 1), equally likely
    let p1 = class_posterior(1.0, 1.0, -1.0, 1.0, 0.5);
    let p2 = class_posterior(2.0, 1.0, -1.0, 1.0, 0.5);
    assert!(close(p1, 0.881, 0.001) && close(p2, 0.982, 0.001), "B2a: got p(A|1) = {p1:.3}, p(A|2) = {p2:.3}, expected 0.881 and 0.982");
    // B2b: the sharpened density at 2 against at 1
    for (w, want) in [(0.0, 0.61), (4.0, 0.94), (8.0, 1.45)] {
        let ratio = gauss(2.0, 1.0, 1.0) / gauss(1.0, 1.0, 1.0) * (p2 / p1).powf(w);
        assert!(close(ratio, want, 0.01), "B2b, w = {w}: got the ratio {ratio:.2}, expected {want}");
    }
    let rare = class_posterior(0.0, 1.0, -1.0, 1.0, 0.1);
    assert!(close(rare, 0.1, 1e-5), "halfway between the classes, p(A) = 0.1: got {rare}, expected 0.1");
}

#[test]
fn c3_sharpened_g4() {
    let (m, s) = sharpened_mean_sd(0.0, -1.0, 1.0, 1.0, 0.5);
    assert!(close(m, -1.0, 0.001) && close(s, 1.0, 0.001), "w = 0: got mean {m:.3}, sd {s:.3}, expected the class itself, −1 and 1");
    let (m, s) = sharpened_mean_sd(3.0, -1.0, 1.0, 1.0, 0.5);
    assert!(close(m, -1.54, 0.01) && close(s, 0.77, 0.01), "w = 3: got mean {m:.3}, sd {s:.3}, expected −1.54 and 0.77 (G2, G4)");
}

#[test]
fn c4_unguided_sampling_hits_the_class() {
    let levels: Vec<f32> = (0..=50).map(|k| cosine_alpha_bar(1.0 - k as f32 / 50.0)).collect();
    let class_a = [(1.0, -1.0, 1.0)];
    let both = [(0.5, -1.0, 1.0), (0.5, 1.0, 1.0)];
    let mut rng = Rng::new(1);
    let starts: Vec<f32> = (0..2000).map(|_| rng.normal()).collect();
    for (w, stochastic, m_want, s_want) in [(0.0, false, -1.0, 1.0), (0.0, true, -1.0, 1.0), (-1.0, false, 0.0, 2f32.sqrt())] {
        let mut srng = Rng::new(2);
        let s: Vec<f32> = starts
            .iter()
            .map(|&x| guided_sample(x, &levels, w, |x, ab| eps_mixture(x, ab, &class_a), |x, ab| eps_mixture(x, ab, &both), if stochastic { Some(&mut srng) } else { None }))
            .collect();
        let m = s.iter().sum::<f32>() / 2000.0;
        let sd = (s.iter().map(|v| (v - m).powi(2)).sum::<f32>() / 2000.0).sqrt();
        assert!(close(m, m_want, 0.1) && close(sd, s_want, 0.1), "w = {w}, stochastic {stochastic}: samples have mean {m:.2}, sd {sd:.2}, expected about {m_want} and {s_want:.2}");
    }
}
