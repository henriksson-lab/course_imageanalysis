// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day11 cs_mri::

use day11::cs_mri::*;
use imgcourse::fft::fft2_real;
use imgcourse::Image;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_mask_count_centre_symmetry() {
    let n = 64;
    let mask = variable_density_mask(n, 16, 6, 3.0, 1);
    let count = mask.iter().filter(|m| **m).count();
    assert!((16..=17).contains(&count), "asked for 16 lines (4×): got {count}");
    for k in [0, 1, 2, 3, 61, 62, 63] {
        assert!(mask[k], "the centre (|frequency| ≤ 3) must be measured: line {k} is missing");
    }
    for k in 1..n {
        assert_eq!(mask[k], mask[n - k], "lines {k} and {} (frequencies f and −f) must be taken together", n - k);
    }
}

#[test]
fn c1_mask_denser_at_the_centre() {
    // Over many masks, lines with |f| < 16 are measured more often than lines with |f| ≥ 16.
    let n = 64;
    let (mut inner, mut outer) = (0, 0);
    for seed in 0..50 {
        let mask = variable_density_mask(n, 16, 4, 3.0, seed);
        for k in 1..n / 2 {
            if mask[k] && k > 2 {
                if k < 16 { inner += 1 } else { outer += 1 }
            }
        }
    }
    assert!(inner > 3 * outer, "q = 3 should favour low frequencies: {inner} lines drawn below |f| = 16, {outer} above");
    // q = 0: every line equally likely.
    let mask = variable_density_mask(n, 64, 0, 0.0, 3);
    assert!(mask.iter().all(|m| *m), "asking for all 64 lines must give all of them");
}

/// D2: an 8-row image whose column 0 is the profile (0, 0, 5, 0, 0, 0, 0, 0); the other
/// columns are 0. Lines are rows of k-space, so the fold is along y.
fn d2() -> Image<f32> {
    Image::from_fn(4, 8, |x, y| if x == 0 && y == 2 { 5.0 } else { 0.0 })
}

#[test]
fn c2_fold_over_d2() {
    let even: Vec<bool> = (0..8).map(|k| k % 2 == 0).collect();
    let z = zero_filled(&d2(), &even);
    let expected = [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 2.5, 0.0];
    for y in 0..8 {
        assert!(close(z[(0, y)], expected[y], 1e-4), "D2(a), every 2nd line: row {y}: got {}, expected {}", z[(0, y)], expected[y]);
        assert!(close(z[(1, y)], 0.0, 1e-4), "D2(a): column 1 should stay 0, row {y} got {}", z[(1, y)]);
    }
    let fourth: Vec<bool> = (0..8).map(|k| k % 4 == 0).collect();
    let z = zero_filled(&d2(), &fourth);
    let expected = [1.25, 0.0, 1.25, 0.0, 1.25, 0.0, 1.25, 0.0];
    for y in 0..8 {
        assert!(close(z[(0, y)], expected[y], 1e-4), "D2(b), every 4th line: row {y}: got {}, expected {}", z[(0, y)], expected[y]);
    }
}

#[test]
fn c3_data_consistency() {
    let truth = Image::from_fn(16, 16, |x, y| ((x * 3 + y * 5) % 7) as f32 / 7.0);
    let kspace = fft2_real(&truth);
    let all = vec![true; 16];
    let out = data_consistency(&Image::new(16, 16, 0.0), &kspace, &all);
    for (x, y) in out.coords() {
        assert!(close(out[(x, y)], truth[(x, y)], 1e-4), "all lines measured: the result must be the truth, ({x}, {y}) got {}", out[(x, y)]);
    }
    let none = vec![false; 16];
    let x0 = Image::from_fn(16, 16, |x, y| (x + y) as f32 / 30.0);
    let out = data_consistency(&x0, &kspace, &none);
    for (x, y) in out.coords() {
        assert!(close(out[(x, y)], x0[(x, y)], 1e-4), "no lines measured: x must not change, ({x}, {y}) got {}", out[(x, y)]);
    }
    // Some lines: afterwards those lines of k-space agree with the data.
    let some: Vec<bool> = (0..16).map(|k| [0, 1, 4, 12, 15].contains(&k)).collect(); // pairs k, 16 − k
    let k_out = fft2_real(&data_consistency(&x0, &kspace, &some));
    for (kx, ky) in k_out.coords() {
        if some[ky] {
            assert!((k_out[(kx, ky)] - kspace[(kx, ky)]).abs() < 1e-3, "measured line {ky} must equal the data after the step");
        }
    }
}

#[test]
fn c4_cs_beats_zero_filling() {
    // A piecewise-constant 64 × 64 image (sparse in Haar), 3× variable-density sampling.
    let n = 64;
    let truth = Image::from_fn(n, n, |x, y| {
        let (u, v) = (x as f32 - 30.0, y as f32 - 34.0);
        if u * u / 500.0 + v * v / 300.0 < 1.0 { if x > 36 && y < 34 { 1.0 } else { 0.6 } } else { 0.1 }
    });
    let mask = variable_density_mask(n, n / 3, 8, 3.0, 2);
    let mut kspace = fft2_real(&truth);
    for (kx, ky) in kspace.coords() {
        if !mask[ky] {
            kspace[(kx, ky)] = imgcourse::Complex::ZERO;
        }
    }
    let err = |img: &Image<f32>| img.zip_map(&truth, |a, b| (a - b) * (a - b)).mean().sqrt();
    let zf = err(&zero_filled(&truth, &mask));
    let cs = err(&cs_reconstruct(&kspace, &mask, 0.01, 3, 100));
    assert!(cs < 0.7 * zf, "CS should clearly beat zero filling: RMS error {cs:.4} (CS) against {zf:.4} (zero filled)");
    // λ = 0 with all lines measured: the truth itself.
    let full = fft2_real(&truth);
    let rec = cs_reconstruct(&full, &vec![true; n], 0.0, 3, 3);
    assert!(err(&rec) < 1e-4, "all lines and λ = 0: the result must be the truth (RMS error {})", err(&rec));
}
