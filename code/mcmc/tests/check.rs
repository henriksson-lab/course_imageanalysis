// Checks for the coding exercises, using the numbers of the pen-and-paper exercises and the
// exact probabilities of the 3 × 3 torus (listed here, as in the Ising unit).
// Run: cargo test -p mcmc

use imgcourse::{Image, Rng};
use mcmc::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

/// A 3 × 3 lattice whose centre (1, 1) has the neighbours right, left, below, above.
fn centre_with(neighbours: [i8; 4]) -> Spins {
    let mut s = Image::new(3, 3, 1i8);
    s[(2, 1)] = neighbours[0];
    s[(0, 1)] = neighbours[1];
    s[(1, 2)] = neighbours[2];
    s[(1, 0)] = neighbours[3];
    s
}

#[test]
fn c1_local_field_b1() {
    let p = |u: f32, t: f32| 1.0 / (1.0 + (-2.0 * u / t).exp());
    let u = local_field(&centre_with([1, 1, 1, -1]), 1, 1, 1.0, 0.0);
    assert!(close(u, 2.0, 1e-6), "B1(a), neighbours (+, +, +, −): got u = {u}, expected 2");
    assert!(close(p(u, 2.0), 0.881, 0.001), "B1(a): p(+1) at T = 2 should be 0.881");
    let u = local_field(&centre_with([1, 1, -1, -1]), 1, 1, 1.0, 0.0);
    assert!(close(u, 0.0, 1e-6), "B1(b), (+, +, −, −): got u = {u}, expected 0");
    let u = local_field(&centre_with([1, 1, -1, -1]), 1, 1, 1.0, 0.5);
    assert!(close(u, 0.5, 1e-6), "B1(c), with h = 0.5: got u = {u}, expected 0.5");
    // periodic: the corner (0, 0) of an all-(+1) lattice still has 4 neighbours
    let u = local_field(&Image::new(3, 3, 1i8), 0, 0, 1.0, 0.0);
    assert!(close(u, 4.0, 1e-6), "corner pixel of a periodic lattice: got u = {u}, expected 4");
}

/// Exact probabilities of the energy levels −18, −10, −6, −2, 2, 6 of the 3 × 3 torus (J = 1).
fn exact_levels(t: f32) -> [f32; 6] {
    let levels = [-18.0f32, -10.0, -6.0, -2.0, 2.0, 6.0];
    let counts = [2.0f32, 18.0, 48.0, 198.0, 144.0, 102.0]; // Ising unit, E2
    let w: Vec<f32> = levels.iter().zip(counts).map(|(e, c)| c * (-e / t).exp()).collect();
    let z: f32 = w.iter().sum();
    [w[0] / z, w[1] / z, w[2] / z, w[3] / z, w[4] / z, w[5] / z]
}

/// The fraction of sweeps spent at each energy level, after a burn-in.
fn sampled_levels(sweep: &mut dyn FnMut(&mut Spins, &mut Rng)) -> [f32; 6] {
    let mut rng = Rng::new(7);
    let mut s = random_spins(3, 3, &mut rng);
    let mut count = [0usize; 6];
    let n = 40_000;
    for k in 0..(n + 500) {
        sweep(&mut s, &mut rng);
        if k >= 500 {
            let (e, _) = energy_and_magnetisation(&s, 1.0, 0.0);
            let level = [-18.0, -10.0, -6.0, -2.0, 2.0, 6.0].iter().position(|l| close(9.0 * e, *l, 1e-3)).expect("an energy of the 3 × 3 torus");
            count[level] += 1;
        }
    }
    count.map(|c| c as f32 / n as f32)
}

#[test]
fn c2_gibbs_samples_the_3x3_torus() {
    let t = 3.0;
    let got = sampled_levels(&mut |s, rng| gibbs_sweep(s, 1.0, 0.0, t, rng));
    let exact = exact_levels(t);
    for k in 0..6 {
        assert!(close(got[k], exact[k], 0.02), "Gibbs at T = 3: level fractions {got:.3?}, exact {exact:.3?}");
    }
}

#[test]
fn c3_metropolis_samples_the_3x3_torus() {
    let t = 3.0;
    let got = sampled_levels(&mut |s, rng| {
        metropolis_sweep(s, 1.0, 0.0, t, rng);
    });
    let exact = exact_levels(t);
    for k in 0..6 {
        assert!(close(got[k], exact[k], 0.02), "Metropolis at T = 3: level fractions {got:.3?}, exact {exact:.3?}");
    }
}

#[test]
fn c3_metropolis_downhill_always_b2() {
    // B2(b): a −1 among +1 neighbours flips back for sure (ΔE < 0), even when very cold;
    // and nothing else flips (every other flip is uphill by 8 or more at T = 0.01).
    let mut s = Image::new(4, 4, 1i8);
    s[(2, 1)] = -1;
    let accepted = metropolis_sweep(&mut s, 1.0, 0.0, 0.01, &mut Rng::new(1));
    assert_eq!(accepted, 1, "one sweep at T = 0.01: got {accepted} accepted flips, expected 1");
    assert!(s.pixels().iter().all(|&v| v == 1), "after the sweep every spin should be +1:\n{s}");
}

#[test]
fn c4_energy_and_magnetisation() {
    let (e, m) = energy_and_magnetisation(&Image::new(8, 8, 1i8), 1.0, 0.0);
    assert!(close(e, -2.0, 1e-6) && close(m, 1.0, 1e-6), "all +1: got E/n = {e}, m = {m}, expected −2 and 1");
    let checker = Image::from_fn(4, 4, |x, y| if (x + y) % 2 == 0 { 1i8 } else { -1 });
    let (e, m) = energy_and_magnetisation(&checker, 1.0, 0.0);
    assert!(close(e, 2.0, 1e-6) && close(m, 0.0, 1e-6), "checkerboard: got E/n = {e}, m = {m}, expected 2 and 0");
    let (e, _) = energy_and_magnetisation(&Image::new(3, 3, -1i8), 1.0, 0.5);
    assert!(close(e, (-18.0 + 4.5) / 9.0, 1e-6), "all −1 with h = 0.5: got E/n = {e}, expected −13.5/9");
}

#[test]
fn c5_anneal_schedule_b6() {
    let mut rng = Rng::new(3);
    let mut s = random_spins(16, 16, &mut rng);
    let temps = anneal(&mut s, 1.0, 4.0, 0.5, 101, &mut rng);
    assert_eq!(temps.len(), 101, "one temperature per sweep");
    assert!(close(temps[0], 4.0, 1e-4) && close(temps[100], 0.5, 1e-3), "first and last temperature: got {} and {}, expected 4 and 0.5", temps[0], temps[100]);
    let r = temps[1] / temps[0];
    assert!(close(r, 0.979, 0.001), "B6(a): got the factor r = {r:.4}, expected 0.979");
    let (e, _) = energy_and_magnetisation(&s, 1.0, 0.0);
    assert!(e < -1.5, "after cooling to T = 0.5 the energy per spin should be near −2, got {e}");
}
