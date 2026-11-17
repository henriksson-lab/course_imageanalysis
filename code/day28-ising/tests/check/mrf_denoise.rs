// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// The three pixels y = (1, 0, 1) of B2 are a 3 × 1 image: pixel 1 is (0, 0), pixel 3 is (2, 0).
// Run: cargo test -p day28 mrf_denoise::

use imgcourse::{Image, Rng};
use day28::mrf_denoise::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

fn y101() -> Labels {
    Image::from_rows(&[[1u32, 0, 1]])
}

fn costs() -> Costs {
    flip_noise_costs(&y101(), 0.2)
}

#[test]
fn c1_energy_b2() {
    let expected = [("111", 1.386), ("000", 2.773), ("001", 3.386), ("100", 3.386), ("101", 4.000), ("011", 4.773), ("110", 4.773), ("010", 8.159)];
    for (name, e) in expected {
        let l: Vec<u32> = name.bytes().map(|b| (b - b'0') as u32).collect();
        let labels = Image::from_vec(3, 1, l);
        let got = energy(&labels, &costs(), 2.0);
        assert!(close(got, e, 0.002), "B2(a), labelling {name}: got E = {got:.3}, expected {e}");
    }
}

#[test]
fn c1_energy_counts_pairs_once() {
    // a 2 × 2 checkerboard has 4 disagreeing pairs; data costs all zero
    let labels = Image::from_rows(&[[0u32, 1], [1, 0]]);
    let zero = Image::new(2, 2, [0.0f32, 0.0]);
    let e = energy(&labels, &zero, 1.0);
    assert!(close(e, 4.0, 1e-6), "2 × 2 checkerboard, λ = 1: got {e}, expected 4 (4 pairs)");
}

#[test]
fn c2_icm_order_b3() {
    let (l, sweeps) = icm(&y101(), &costs(), 2.0, &[(0, 0), (1, 0), (2, 0)]);
    assert_eq!(l.pixels(), &[0, 0, 0], "B3(a), order 1, 2, 3: got {:?}, expected 000", l.pixels());
    assert_eq!(sweeps, 2, "B3(a): one sweep with changes and one without, got {sweeps} sweeps");
    let (l, _) = icm(&y101(), &costs(), 2.0, &[(1, 0), (0, 0), (2, 0)]);
    assert_eq!(l.pixels(), &[1, 1, 1], "B3(b), order 2, 1, 3: got {:?}, expected 111 (the MAP)", l.pixels());
}

#[test]
fn c2_icm_lambda_zero_is_thresholding() {
    let y = Image::from_rows(&[[1u32, 0, 0, 1], [0, 1, 1, 0]]);
    let order: Vec<(usize, usize)> = y.coords().collect();
    let (l, _) = icm(&y, &flip_noise_costs(&y, 0.2), 0.0, &order);
    assert_eq!(l, y, "λ = 0: ICM should keep the observed image");
}

#[test]
fn c3_gibbs_cold_and_hot() {
    // T → 0 with λ = 2: from 101, a sweep can only go downhill (like ICM, order 1, 2, 3)
    let mut l = y101();
    gibbs_sweep(&mut l, &costs(), 2.0, 0.001, &mut Rng::new(1));
    assert_eq!(l.pixels(), &[0, 0, 0], "T = 0.001: got {:?}, expected 000 (as ICM)", l.pixels());
    // T very large: every label is a coin flip
    let mut ones = 0;
    let mut l = y101();
    let mut rng = Rng::new(2);
    for _ in 0..2000 {
        gibbs_sweep(&mut l, &costs(), 2.0, 1e6, &mut rng);
        ones += l.pixels().iter().filter(|&&v| v == 1).count();
    }
    let frac = ones as f32 / 6000.0;
    assert!(close(frac, 0.5, 0.03), "T = 10^6: got a fraction {frac:.3} of ones, expected about 0.5");
}

#[test]
fn c4_marginals_b4() {
    let p = marginals(&y101(), &costs(), 2.0, 100, 60_000, &mut Rng::new(3));
    assert!(close(p[(0, 0)], 0.747, 0.01), "B4: got p(L1 = 1) = {:.3}, expected 0.747", p[(0, 0)]);
    assert!(close(p[(1, 0)], 0.643, 0.01), "B4: got p(L2 = 1) = {:.3}, expected 0.643", p[(1, 0)]);
    assert!(close(p[(2, 0)], 0.747, 0.01), "B4: got p(L3 = 1) = {:.3}, expected 0.747", p[(2, 0)]);
}

#[test]
fn c5_mean_field_b9() {
    let q = mean_field(&costs(), 2.0, 0);
    assert!(close(q[(0, 0)], 0.8, 1e-3) && close(q[(1, 0)], 0.2, 1e-3), "B9 start, m = tanh(h_p): got q = {:?}, expected (0.8, 0.2, 0.8)", q.pixels());
    let q = mean_field(&costs(), 2.0, 1);
    let expected = [0.5465, 0.4999, 0.8000]; // m = (0.093, −0.0003, 0.600)
    for k in 0..3 {
        assert!(close(q[(k, 0)], expected[k], 0.002), "B9 after one sweep: got q = {:?}, expected {expected:?}", q.pixels());
    }
    let q = mean_field(&costs(), 2.0, 200);
    let expected = [0.9525, 0.9035, 0.9525]; // m = (0.905, 0.807, 0.905)
    for k in 0..3 {
        assert!(close(q[(k, 0)], expected[k], 0.002), "B9 fixed point: got q = {:?}, expected {expected:?}", q.pixels());
    }
}
