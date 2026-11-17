// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p lifetime

use imgcourse::{Rng, Volume};
use lifetime::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// No IRF, a long period (no wrap-around), 500 bins of 0.1 ns.
const IDEAL: Setup = Setup { period: 50.0, bins: 500, irf_delay: 0.0, irf_sigma: 0.0 };

#[test]
fn c1_photons_still_to_come_b1() {
    let counts = simulate(2.0, 100_000, 0.0, &IDEAL, &mut Rng::new(1));
    let total: f32 = counts.iter().sum();
    assert!(close(total, 100_000.0, 0.5), "the histogram holds {total} photons, expected 100000");
    // B1: e^-1, e^-2, e^-3 of the photons arrive later than 2, 4 and 6 ns (bins 20, 40, 60 on).
    for (t, expected) in [(2.0, 0.368), (4.0, 0.135), (6.0, 0.050)] {
        let later: f32 = counts[(t * 10.0) as usize..].iter().sum::<f32>() / total;
        assert!(close(later, expected, 0.01), "later than {t} ns: got a fraction {later:.3}, expected {expected}");
    }
}

#[test]
fn c1_irf_adds_its_delay() {
    let setup = Setup { period: 100.0, bins: 1000, irf_delay: 1.0, irf_sigma: 0.2 };
    let counts = simulate(2.5, 100_000, 0.0, &setup, &mut Rng::new(2));
    let mean = mean_arrival(&counts, setup.period);
    assert!(close(mean, 3.5, 0.05), "τ = 2.5 ns, IRF at 1 ns: mean arrival {mean:.3} ns, expected about 3.5 (means add)");
}

#[test]
fn c1_wraparound_b4_b8() {
    // τ = 8 ns at 80 MHz, no IRF. With wrap-around the first 2 ns hold
    // (1 - e^(-2/8)) / (1 - e^(-12.5/8)) = 0.280 of the photons (B8); without it 0.221.
    let setup = Setup { period: 12.5, bins: 125, irf_delay: 0.0, irf_sigma: 0.0 };
    let counts = simulate(8.0, 200_000, 0.0, &setup, &mut Rng::new(3));
    let first: f32 = counts[..20].iter().sum::<f32>() / counts.iter().sum::<f32>();
    assert!(close(first, 0.280, 0.006), "fraction in the first 2 ns: got {first:.3}, expected 0.280 (did the late photons wrap round?)");
}

#[test]
fn c1_background_is_flat() {
    let counts = simulate(2.0, 50_000, 1.0, &DEMO, &mut Rng::new(4));
    let mean = mean_arrival(&counts, DEMO.period);
    assert!(close(mean, 6.25, 0.1), "only background: mean arrival {mean:.2} ns, expected the middle of the window, 6.25");
}

#[test]
fn c2_mean_arrival_b3() {
    // 125 bins of 0.1 ns. The fluorescence: 900 photons around 3.1 ns (bins centred at 3.05
    // and 3.15). B3(b): add 100 background photons spread evenly, 0.8 per bin.
    let mut counts = vec![0.0f32; 125];
    counts[30] = 450.0;
    counts[31] = 450.0;
    let m = mean_arrival(&counts, 12.5);
    assert!(close(m, 3.1, 1e-4), "B3(a): mean arrival {m}, expected 3.1");
    assert!(close(m - 0.6, 2.5, 1e-4), "B3(a): lifetime {}, expected 2.5", m - 0.6);
    for c in counts.iter_mut() {
        *c += 0.8;
    }
    let m = mean_arrival(&counts, 12.5);
    assert!(close(m, 3.415, 1e-3), "B3(b): with 10 % background the mean arrival is {m}, expected 3.415");
}

#[test]
fn c3_two_gates_b2() {
    // Bins of 1 ns; the gates 0–2 ns and 2–4 ns hold 632 and 233 photons.
    let mut counts = vec![0.0f32; 12];
    counts[..4].copy_from_slice(&[400.0, 232.0, 150.0, 83.0]);
    let tau = two_gates(&counts, 12.0, 0.0, 2.0);
    assert!(close(tau, 2.0, 0.01), "B2(a): got τ = {tau}, expected 2.0");
    for (per_gate, expected) in [(50.0, 2.27), (100.0, 2.54)] {
        let with_bg: Vec<f32> = counts.iter().map(|c| c + per_gate / 2.0).collect();
        let tau = two_gates(&with_bg, 12.0, 0.0, 2.0);
        assert!(close(tau, expected, 0.01), "B2(b): {per_gate} background photons per gate: got τ = {tau:.3}, expected {expected}");
    }
}

#[test]
fn c3_two_gates_unbiased_by_wraparound_b10() {
    let setup = Setup { period: 12.5, bins: 125, irf_delay: 0.0, irf_sigma: 0.0 };
    let counts = simulate(8.0, 400_000, 0.0, &setup, &mut Rng::new(5));
    let tau = two_gates(&counts, setup.period, 1.0, 4.0);
    assert!(close(tau, 8.0, 0.25), "τ = 8 ns at 80 MHz: two gates give {tau:.2}, expected about 8 (B10c)");
}

#[test]
fn c4_lifetime_image_bins_and_masks() {
    // A 3 × 3 image whose pixels all have the same histogram shape: 10 bins of 1 ns, mean 2.5 ns.
    let shape = [0.0, 3.0, 3.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let m = mean_arrival(&shape, 10.0);
    let mut stack = Volume::from_fn(3, 3, 10, |_, _, j| shape[j]);
    for j in 0..10 {
        stack[(2, 2, j)] = 0.0; // an empty pixel
    }
    let irf = 0.5;
    let single = lifetime_image(&stack, 10.0, irf, 0, 5.0);
    assert!(close(single[(0, 0)], m - irf, 1e-4), "pixel (0, 0): got {}, expected {}", single[(0, 0)], m - irf);
    assert_eq!(single[(2, 2)], 0.0, "the empty pixel (2, 2) must be masked to 0");
    // With 3 × 3 binning the empty pixel borrows its neighbours' photons.
    let binned = lifetime_image(&stack, 10.0, irf, 1, 5.0);
    assert!(close(binned[(2, 2)], m - irf, 1e-4), "binned pixel (2, 2): got {}, expected {}", binned[(2, 2)], m - irf);
    // A mask threshold above 10 photons: single pixels are masked, binned ones (up to 90) are not.
    let masked = lifetime_image(&stack, 10.0, irf, 0, 11.0);
    assert_eq!(masked[(1, 1)], 0.0, "10 photons with min_photons = 11 must be masked");
    let corner = lifetime_image(&stack, 10.0, irf, 1, 35.0);
    assert!(corner[(0, 0)] > 0.0, "the corner (0, 0) bins 4 pixels (40 photons), so min_photons = 35 keeps it");
    assert_eq!(corner[(2, 2)], 0.0, "the corner (2, 2) bins 3 full pixels (30 photons), so min_photons = 35 masks it");
}
