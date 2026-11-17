// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day11 dictionaries::

use day11::dictionaries::*;
use imgcourse::Image;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// The six atoms of F1.
fn f1_dict() -> Vec<Vec<f64>> {
    let r = std::f64::consts::FRAC_1_SQRT_2;
    vec![
        vec![0.5, 0.5, 0.5, 0.5],
        vec![0.5, 0.5, -0.5, -0.5],
        vec![0.5, -0.5, 0.5, -0.5],
        vec![0.5, -0.5, -0.5, 0.5],
        vec![r, r, 0.0, 0.0],
        vec![0.0, 0.0, r, r],
    ]
}

#[test]
fn c1_omp_f1() {
    let y = [3.0, 3.0, 1.0, 1.0];
    let c = omp(&f1_dict(), &y, 1, 0.0);
    assert!(close(c[4], 3.0 * 2f64.sqrt(), 1e-9) && c.iter().filter(|v| **v != 0.0).count() == 1,
        "F1(a): one atom: expected atom 5 with coefficient 4.243, got {c:?}");
    let c = omp(&f1_dict(), &y, 2, 0.0);
    assert!(close(c[4], 3.0 * 2f64.sqrt(), 1e-9) && close(c[5], 2f64.sqrt(), 1e-9) && c.iter().filter(|v| **v != 0.0).count() == 2,
        "F1(b): two atoms: expected atoms 5 and 6 with (4.243, 1.414), got {c:?}");
    let r: f64 = y.iter().zip(rebuild(&f1_dict(), &c)).map(|(a, b)| (a - b).powi(2)).sum();
    assert!(r < 1e-12, "F1(b): the residual should be 0, got ‖r‖² = {r}");
    // The error target: stop as soon as ‖r‖² ≤ 2.5 (after atom 5 it is 2).
    let c = omp(&f1_dict(), &y, 4, 2.5);
    assert_eq!(c.iter().filter(|v| **v != 0.0).count(), 1, "with max_error 2.5, OMP should stop after one atom: {c:?}");
}

#[test]
fn c2_update_atom_f2() {
    let residuals = vec![vec![2.0, 1.0], vec![4.0, 2.0], vec![-1.0, -1.0]];
    let (d, c) = update_atom(&residuals, &[1.0, 0.0]);
    let s = d[0].signum(); // the sign of an atom is free
    assert!(close(s * d[0], 0.884, 1e-3) && close(s * d[1], 0.467, 1e-3), "F2(a): atom got {d:?}, expected ±(0.884, 0.467)");
    let expected = [2.24, 4.47, -1.35];
    for (g, e) in c.iter().zip(expected) {
        assert!(close(s * g, e, 0.01), "F2(b): coefficients got {c:?}, expected ±{expected:?}");
    }
    let err: f64 = residuals.iter().zip(&c).map(|(r, ci)| (0..2).map(|p| (r[p] - ci * d[p]).powi(2)).sum::<f64>()).sum();
    assert!(close(err, 0.19, 0.01), "F2(b): remaining squared error got {err}, expected 0.19 (the small eigenvalue)");
}

#[test]
fn c3_ksvd_learns_three_directions() {
    // The k-means slide: 2D points along three directions through 0, on both sides.
    let angles = [10f64, 70.0, 130.0].map(|a| a.to_radians());
    let mut patches = Vec::new();
    for i in 0..60 {
        let t = angles[i % 3];
        let c = (0.5 + (i * 7 % 10) as f64 / 6.0) * if i % 2 == 0 { 1.0 } else { -1.0 };
        patches.push(vec![c * t.cos(), c * t.sin()]);
    }
    let start: Vec<Vec<f64>> = [0f64, 60.0, 120.0].iter().map(|a| vec![a.to_radians().cos(), a.to_radians().sin()]).collect();
    let (dict, errors) = ksvd(&patches, &start, 1, 10);
    assert_eq!(errors.len(), 10, "one error per iteration");
    assert!(errors[9] < 1e-6, "one atom per direction fits every point: error after 10 iterations {:e}, errors {errors:?}", errors[9]);
    for d in &dict {
        assert!(close(d[0] * d[0] + d[1] * d[1], 1.0, 1e-9), "atoms must have length 1, got {d:?}");
    }
}

#[test]
fn c4_denoise_exact_dictionary() {
    let img = Image::from_fn(6, 5, |x, y| ((x * 3 + y * 5) % 7) as f32 / 7.0);
    // The 4 pixel atoms of 2 × 2 patches: every patch is coded exactly, so nothing changes.
    let pixels: Vec<Vec<f64>> = (0..4).map(|j| (0..4).map(|i| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    let out = denoise(&img, &pixels, 0.0, 0.5);
    for (x, y) in img.coords() {
        assert!((out[(x, y)] - img[(x, y)]).abs() < 1e-5, "exact dictionary: pixel ({x}, {y}) got {}, expected {}", out[(x, y)], img[(x, y)]);
    }
}

#[test]
fn c4_denoise_averages() {
    // Only the constant atom: each patch becomes its mean; an interior pixel of a 2 × 2 patch
    // grid is covered by 4 patches (F4: 64 for 8 × 8).
    let img = Image::from_rows(&[[0.0f32, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0]]);
    let constant = vec![vec![0.5; 4]];
    let out = denoise(&img, &constant, 0.0, 0.0);
    // every 2 × 2 patch contains the centre pixel once: mean 1
    assert!(close(out[(1, 1)] as f64, 1.0, 1e-5), "centre: the average of 4 patch means (each 1): got {}", out[(1, 1)]);
    assert!(close(out[(0, 0)] as f64, 1.0, 1e-5), "corner: covered by one patch with mean 1: got {}", out[(0, 0)]);
    let out = denoise(&img, &constant, 0.0, 4.0);
    assert!(close(out[(1, 1)] as f64, (4.0 * 4.0 + 4.0) / 8.0, 1e-5), "centre with λ = 4: (λ·4 + 4·1)/(λ + 4) = 2.5, got {}", out[(1, 1)]);
}
