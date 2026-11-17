// Checks for the coding exercises, using the numbers of the lecture.
// Run: cargo test -p images pca::

use imgcourse::samples;
use images::pca::*;

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-4
}

/// The four two-pixel images of the lecture.
const FOUR: [[f32; 2]; 4] = [[0.2, 0.4], [0.4, 0.2], [0.6, 0.8], [0.8, 0.6]];

#[test]
fn c1_mean_and_covariance() {
    let (mean, s) = mean_and_covariance(&FOUR);
    assert!(close(mean[0], 0.5) && close(mean[1], 0.5), "mean: got {mean:?}, expected [0.5, 0.5]");
    let expected = [[0.05, 0.03], [0.03, 0.05]];
    for i in 0..2 {
        for j in 0..2 {
            assert!(close(s[i][j], expected[i][j]), "S: got {s:?}, expected {expected:?}");
        }
    }
}

#[test]
fn c2_eigen2() {
    let ([l1, l2], [u1, u2]) = eigen2([[0.05, 0.03], [0.03, 0.05]]);
    assert!(close(l1, 0.08) && close(l2, 0.02), "eigenvalues: got {l1}, {l2}, expected 0.08, 0.02");
    let r = std::f32::consts::FRAC_1_SQRT_2;
    assert!(close(u1[0].abs(), r) && close(u1[1].abs(), r) && u1[0] * u1[1] > 0.0, "PC1 should be ±(1, 1)/√2, got {u1:?}");
    assert!(close(u2[0].abs(), r) && close(u2[1].abs(), r) && u2[0] * u2[1] < 0.0, "PC2 should be ±(1, −1)/√2, got {u2:?}");
    // A diagonal matrix: the axes are the principal components.
    let ([l1, l2], [u1, _]) = eigen2([[0.01, 0.0], [0.0, 0.04]]);
    assert!(close(l1, 0.04) && close(l2, 0.01) && close(u1[1].abs(), 1.0), "diag(0.01, 0.04): got {l1}, {l2}, {u1:?}");
}

#[test]
fn c3_variance_plus_error_is_constant() {
    let (mean, s) = mean_and_covariance(&FOUR);
    let total = s[0][0] + s[1][1];
    for degrees in [0.0f32, 30.0, 45.0, 100.0] {
        let a = degrees.to_radians();
        let (scores, err) = project(&FOUR, mean, [a.cos(), a.sin()]);
        let var = scores.iter().map(|s| s * s).sum::<f32>() / scores.len() as f32;
        assert!(close(var + err, total), "at {degrees}°: variance {var} + error {err} should be {total}");
    }
    let (_, err) = project(&FOUR, mean, [std::f32::consts::FRAC_1_SQRT_2; 2]);
    assert!(close(err, 0.02), "along PC1 the error should be λ2 = 0.02, got {err}");
}

#[test]
fn c4_patches_and_power_iteration() {
    let p = patches(&samples::cell(), 3, 4);
    assert_eq!(p.len(), 64 * 64, "3 × 3 patches every 4 pixels in a 256 × 256 image");
    assert_eq!(p[0].len(), 9);
    let (lambda, u) = power_iteration(&[vec![0.05, 0.03], vec![0.03, 0.05]], 100);
    assert!(close(lambda, 0.08) && close(u[0], u[1]), "power iteration: got {lambda}, {u:?}");
    // PC1 of small patches of a smooth image is (close to) 'all pixels together': overall brightness.
    let (_, s) = covariance(&p);
    let (_, pc1) = power_iteration(&s, 200);
    assert!(pc1.iter().all(|&v| v * pc1[4] > 0.0), "PC1 of 3 × 3 patches should have one sign everywhere, got {pc1:?}");
}
