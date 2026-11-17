// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p flim decayfit::

use flim::decayfit::*;
use imgcourse::Rng;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// A Gaussian IRF of width `sigma` at `delay`, as a histogram of `n` bins over `period`, sum 1.
fn gaussian_irf(n: usize, period: f32, delay: f32, sigma: f32) -> Vec<f32> {
    let dt = period / n as f32;
    let g: Vec<f32> = (0..n).map(|i| (-((i as f32 + 0.5) * dt - delay).powi(2) / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = g.iter().sum();
    g.iter().map(|v| v / s).collect()
}

/// A Poisson random count with mean `mu`.
fn poisson(mu: f32, rng: &mut Rng) -> f32 {
    if mu > 50.0 {
        return (mu + mu.sqrt() * rng.normal()).round().max(0.0);
    }
    let (limit, mut k, mut p) = ((-mu).exp(), 0.0, 1.0);
    loop {
        p *= rng.uniform();
        if p < limit {
            return k;
        }
        k += 1.0;
    }
}

#[test]
fn c1_bin_fractions_b4() {
    // No IRF (a spike in bin 0), bins of 2 ns over a long period (no wrap-around to speak of).
    let mut spike = vec![0.0; 50];
    spike[0] = 1.0;
    for (tau, expected) in [(1.0, [0.865, 0.117, 0.018]), (3.0, [0.487, 0.250, 0.194])] {
        let mu = model(tau, 1.0, 0.0, &spike, 100.0);
        let got = [mu[0], mu[1], mu[2] + mu[3]];
        for g in 0..3 {
            assert!(close(got[g], expected[g], 0.002), "τ = {tau} ns, gates 0–2, 2–4, 4–8 ns: got {got:?}, expected {expected:?} (B4)");
        }
    }
}

#[test]
fn c1_photons_background_and_irf_shift() {
    let irf = gaussian_irf(64, 12.5, 1.0, 0.15);
    let mu = model(2.5, 1000.0, 2.0, &irf, 12.5);
    let total: f32 = mu.iter().sum();
    assert!(close(total, 1000.0 + 2.0 * 64.0, 0.05), "sum of the model: got {total}, expected 1000 photons + 2 per bin × 64 bins");
    // A spike IRF in bin 3 shifts the whole decay by 3 bins.
    let (mut spike0, mut spike3) = (vec![0.0; 64], vec![0.0; 64]);
    spike0[0] = 1.0;
    spike3[3] = 1.0;
    let (a, b) = (model(2.5, 1.0, 0.0, &spike0, 12.5), model(2.5, 1.0, 0.0, &spike3, 12.5));
    for i in 0..64 {
        assert!(close(b[(i + 3) % 64], a[i], 1e-6), "an IRF in bin 3 must shift the decay by 3 bins (bin {i})");
    }
    // Wrap-around: the periodic decay of τ = 8 ns at 12.5 ns still has photons in the last bin.
    let wrapped = model(8.0, 1.0, 0.0, &spike0, 12.5);
    let s: f32 = wrapped.iter().sum();
    assert!(close(s, 1.0, 1e-4), "the periodic decay must hold all the photons in one period: sum {s}");
}

#[test]
fn c2_three_costs() {
    let (y, mu) = ([0.0, 2.0, 4.0], [1.0, 2.0, 2.0]);
    let ls = cost(&y, &mu, Cost::LeastSquares);
    let ney = cost(&y, &mu, Cost::Neyman);
    let poi = cost(&y, &mu, Cost::Poisson);
    assert!(close(ls, 5.0, 1e-5), "least squares: got {ls}, expected 1 + 0 + 4 = 5");
    assert!(close(ney, 2.0, 1e-5), "Neyman: got {ney}, expected 1/1 + 0/2 + 4/4 = 2 (an empty bin gets weight 1)");
    let expected = 2.0 * (1.0 + 4.0 * 2f32.ln() - 2.0);
    assert!(close(poi, expected, 1e-4), "Poisson deviance: got {poi}, expected 2[(0 + 1) + 0 + (4 ln 2 − 2)] = {expected}");
    assert!(close(cost(&mu, &mu, Cost::Poisson), 0.0, 1e-6), "the deviance of a perfect model must be 0");
}

#[test]
fn c3_fit_noise_free_model() {
    let irf = gaussian_irf(64, 12.5, 1.0, 0.15);
    let y = model(2.5, 10_000.0, 5.0, &irf, 12.5);
    for kind in [Cost::LeastSquares, Cost::Neyman, Cost::Poisson] {
        let f = fit(&y, &irf, 12.5, kind, 1.5);
        assert!(close(f.tau, 2.5, 0.01), "{kind:?}: fitted τ {}, expected 2.5 (no noise, start 1.5)", f.tau);
        assert!(close(f.photons, 10_000.0, 20.0) && close(f.background, 5.0, 0.1), "{kind:?}: photons {} and background {}, expected 10000 and 5", f.photons, f.background);
    }
}

#[test]
fn c3_poisson_fit_near_the_photon_limit() {
    // 200 noisy histograms of 1000 photons: the Poisson fit should be unbiased, with a spread
    // not far above the ideal τ/√N = 0.079 ns. (F is about 1.5 here: the fit also estimates the
    // background, which costs some precision; with the background fixed F is about 1.2.)
    let irf = gaussian_irf(64, 12.5, 1.0, 0.15);
    let mu = model(2.5, 1000.0, 0.2, &irf, 12.5); // and 0.2 background counts per bin
    let mut rng = Rng::new(7);
    let taus: Vec<f32> = (0..200)
        .map(|_| {
            let y: Vec<f32> = mu.iter().map(|&m| poisson(m, &mut rng)).collect();
            fit(&y, &irf, 12.5, Cost::Poisson, 2.0).tau
        })
        .collect();
    let mean = taus.iter().sum::<f32>() / 200.0;
    let sd = (taus.iter().map(|t| (t - mean).powi(2)).sum::<f32>() / 199.0).sqrt();
    assert!(close(mean, 2.5, 0.03), "mean of 200 Poisson fits: {mean:.3} ns, expected about 2.5");
    let f_value = sd / 2.5 * 1000f32.sqrt();
    assert!((0.9..1.8).contains(&f_value), "spread {sd:.3} ns, F = {f_value:.2}: expected F between 0.9 and 1.8");
}

#[test]
fn c4_mean_lifetimes_b3() {
    let (ta, tf) = mean_lifetimes(&[0.7, 0.3], &[1.0, 4.0]);
    assert!(close(ta, 1.90, 0.005) && close(tf, 2.89, 0.005), "B3(b): got τa = {ta:.3}, τf = {tf:.3}, expected 1.90 and 2.89");
    let (ta, tf) = mean_lifetimes(&[0.7, 0.3], &[1.0, 3.0]);
    assert!(close(ta, 1.60, 0.005) && close(tf, 2.125, 0.005), "B3(c): got τa = {ta:.3}, τf = {tf:.3}, expected 1.60 and 2.13");
    let (ta, tf) = mean_lifetimes(&[0.5, 0.5], &[0.5, 3.0]);
    assert!(close(ta, 1.75, 0.005) && close(tf, 2.64, 0.005), "lecture example: got τa = {ta:.3}, τf = {tf:.3}, expected 1.75 and 2.64");
}

#[test]
fn c5_global_amplitudes_b4() {
    let shapes = vec![vec![0.865, 0.117, 0.018], vec![0.487, 0.250, 0.194]];
    let a = amplitudes(&[612.0, 203.0, 148.0], &shapes);
    assert!(close(a[0], 315.0, 2.0) && close(a[1], 695.0, 2.0), "B4(b): got a = ({:.1}, {:.1}), expected (315, 695)", a[0], a[1]);
    // Exact data are recovered exactly.
    let y: Vec<f32> = (0..3).map(|j| 100.0 * shapes[0][j] + 400.0 * shapes[1][j]).collect();
    let a = amplitudes(&y, &shapes);
    assert!(close(a[0], 100.0, 0.05) && close(a[1], 400.0, 0.05), "exact mixture of 100 and 400: got {a:?}");
}
