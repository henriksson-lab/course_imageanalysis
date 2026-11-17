// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day10 bilateral_nlm::

use day10::bilateral_nlm::*;
use imgcourse::{samples, Image};

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_effective_number_b3() {
    let v = [1.0, 2.0, 3.0, 4.0, 5.0];
    for (w, expected) in [(vec![1.0, 1.0, 1.0, 1.0], 4.0), (vec![1.0, 0.1, 0.1, 0.1], 1.69 / 1.03), (vec![1.0, 0.5, 0.5, 0.5, 0.5], 4.5)] {
        let (_, n) = weighted_mean(&v[..w.len()], &w);
        assert!(close(n, expected, 1e-3), "weights {w:?}: N_eff got {n}, expected {expected:.2}");
    }
    let (m, _) = weighted_mean(&[10.0, 20.0], &[3.0, 1.0]);
    assert!(close(m, 12.5, 1e-4), "values 10, 20 with weights 3, 1: mean got {m}, expected 12.5");
}

#[test]
fn c2_bilateral_b1() {
    let img = Image::from_rows(&[[12.0, 10.0, 50.0]]);
    let out = bilateral(&img, 1e6, 10.0, 1);
    assert!(close(out[(1, 0)], 11.0, 0.01), "B1, the centre pixel: got {}, expected 11.0", out[(1, 0)]);
}

#[test]
fn c2_flat_is_unchanged_and_huge_sigma_r_is_a_blur() {
    let flat = Image::new(6, 5, 0.3);
    let out = bilateral(&flat, 2.0, 0.1, 3);
    assert!(close(out.max(), 0.3, 1e-5) && close(out.min(), 0.3, 1e-5), "a flat image must stay flat");
    // sigma_r huge: only the spatial weights are left, the same for every pixel inside.
    let img = samples::cell().crop(50, 50, 9, 9);
    let out = bilateral(&img, 1e6, 1e6, 1);
    let mean3: f32 = (3..6).flat_map(|y| (3..6).map(move |x| (x, y))).map(|p| img[p]).sum::<f32>() / 9.0;
    assert!(close(out[(4, 4)], mean3, 1e-5), "sigma_s and sigma_r huge: got {}, expected the 3 × 3 mean {mean3}", out[(4, 4)]);
}

#[test]
fn c3_nlm_by_hand_b2() {
    let x = [11.0, 9.0, 30.0, 10.0, 12.0, 29.0, 9.0, 11.0];
    let a = nlm_1d(&x, 3, 1, 10.0);
    assert!(close(a, 10.93, 0.01), "B2 (b), h = 10: got {a}, expected 10.93");
    let c = nlm_1d(&x, 3, 1, 1.0);
    assert!(close(c, 9.73, 0.01), "B2 (c), h = 1: got {c}, expected 9.73");
}

#[test]
fn c4_nlm_finds_copies() {
    // A noisy lattice of dots with period 6: NLM must clean it much better than its own noise.
    let truth = Image::from_fn(36, 36, |x, y| if x % 6 < 2 && y % 6 < 2 { 0.8 } else { 0.2 });
    let noisy = samples::add_noise(&truth, 0.1, 7);
    let out = nlm(&noisy, 1, 7, 0.15);
    let rms = |a: &Image<f32>| (a.zip_map(&truth, |p, q| (p - q) * (p - q)).mean()).sqrt();
    assert!(rms(&out) < 0.5 * rms(&noisy), "NLM on the lattice: error {:.3}, noisy {:.3}; expected below half", rms(&out), rms(&noisy));
    let flat = nlm(&Image::new(8, 8, 0.4), 1, 2, 0.1);
    assert!(close(flat.max(), 0.4, 1e-5) && close(flat.min(), 0.4, 1e-5), "a flat image must stay flat");
}
