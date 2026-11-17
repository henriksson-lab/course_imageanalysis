// Checks for the coding exercises, using the numbers of the pen-and-paper exercises (A2 to A6).
// Run: cargo test -p keypoints

use imgcourse::Image;
use keypoints::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_level_sigma_a3() {
    // A3: the six blurred images of octave 0 with σ0 = 1.6, S = 3.
    let expected = [1.60, 2.02, 2.54, 3.20, 4.03, 5.08];
    for (level, e) in expected.iter().enumerate() {
        let got = level_sigma(1.6, 3, 0, level as f32);
        assert!(close(got, *e, 0.006), "octave 0, level {level}: got σ = {got}, expected {e}");
    }
    // A2: σ = 6.0 is octave 1, level 2.72.
    let got = level_sigma(1.6, 3, 1, 2.72);
    assert!(close(got, 6.0, 0.01), "octave 1, level 2.72: got σ = {got}, expected 6.0");
}

#[test]
fn c1_dog_octave_sizes_and_semigroup_b11() {
    let img = imgcourse::samples::cell().crop(60, 60, 48, 48);
    let (blurred, dogs) = dog_octave(&img, 1.6, 3);
    assert_eq!(blurred.len(), 6, "S = 3 needs S + 3 = 6 blurred images, got {}", blurred.len());
    assert_eq!(dogs.len(), 5, "S = 3 needs S + 2 = 5 DoGs, got {}", dogs.len());
    // B11: blurring step by step must agree with blurring once by the total extra σ.
    // L_3 has σ = 3.2: from σ0 = 1.6 that is one blur by sqrt(3.2² − 1.6²).
    let direct = gaussian_blur(&img, (3.2f32 * 3.2 - 1.6 * 1.6).sqrt());
    for (x, y) in [(24, 24), (10, 30), (35, 12)] {
        let (a, b) = (blurred[3][(x, y)], direct[(x, y)]);
        assert!(close(a, b, 3e-3), "L_3 at ({x}, {y}): step by step {a}, in one blur {b}");
    }
    let d = dogs[2][(24, 24)] - (blurred[3][(24, 24)] - blurred[2][(24, 24)]);
    assert!(d.abs() < 1e-6, "D_2 must be L_3 − L_2");
}

#[test]
fn c2_local_extrema_finds_a_planted_peak() {
    // Three DoG levels, all 0.1, with one larger value in the middle level at (4, 3),
    // and one smaller value at (1, 1) of the middle level.
    let mut dogs = vec![Image::new(8, 6, 0.1f32); 3];
    dogs[1][(4, 3)] = 0.5;
    dogs[1][(1, 1)] = -0.3;
    let found = local_extrema(&dogs, 0.05);
    assert!(found.contains(&(4, 3, 1)), "a maximum at (4, 3), level 1: got {found:?}");
    assert!(found.contains(&(1, 1, 1)), "a minimum at (1, 1), level 1: got {found:?}");
    assert_eq!(found.len(), 2, "only the two planted extrema: got {found:?}");
    // Above the threshold only.
    let found = local_extrema(&dogs, 0.4);
    assert_eq!(found, vec![(4, 3, 1)], "threshold 0.4: got {found:?}");
    // A larger value in the level above wins: then (4, 3) is no longer an extremum.
    dogs[2][(5, 3)] = 0.6;
    let found = local_extrema(&dogs, 0.4);
    assert!(found.is_empty(), "(4, 3) has a larger neighbour in the level above: got {found:?}");
}

#[test]
fn c3_parabola_peak_a4() {
    let (offset, value) = parabola_peak(0.6, 1.0, 0.8);
    assert!(close(offset, 1.0 / 6.0, 1e-4), "offset: got {offset}, expected +0.17 (1/6)");
    assert!(close(value, 1.0083, 1e-3), "value: got {value}, expected 1.008");
    let (offset, _) = parabola_peak(0.8, 1.0, 0.6);
    assert!(close(offset, -1.0 / 6.0, 1e-4), "mirrored samples: got {offset}, expected −1/6");
}

/// A 3 × 3 DoG whose finite-difference Hessian at the centre is [dxx, dxy; dxy, dyy]:
/// D = ½ (dxx x² + 2 dxy x y + dyy y²) around the centre.
fn quadratic(dxx: f32, dxy: f32, dyy: f32) -> Image<f32> {
    Image::from_fn(3, 3, |x, y| {
        let (x, y) = (x as f32 - 1.0, y as f32 - 1.0);
        0.5 * (dxx * x * x + 2.0 * dxy * x * y + dyy * y * y)
    })
}

#[test]
fn c4_edge_test_a5() {
    let cases = [
        ("(a) diag(−4, −3)", quadratic(-4.0, 0.0, -3.0), true),
        ("(b) diag(−10, −0.5)", quadratic(-10.0, 0.0, -0.5), false),
        ("(c) diag(3, −2)", quadratic(3.0, 0.0, -2.0), false),
        ("(d) [−6, 2; 2, −1]", quadratic(-6.0, 2.0, -1.0), false),
    ];
    for (name, dog, keep) in cases {
        let got = edge_test(&dog, 1, 1, 10.0);
        assert_eq!(got, keep, "A5 {name}: got keep = {got}, expected {keep}");
    }
}

#[test]
fn c5_orientations_a6() {
    let votes = [(3.0, 42.0), (2.0, 47.0), (1.0, 130.0), (2.5, 44.0), (1.5, 222.0), (2.0, 225.0), (2.6, 228.0), (0.5, 300.0)];
    let hist = orientation_histogram(&votes);
    assert!(close(hist[4], 7.5, 1e-4), "bin 40–49°: got {}, expected 7.5", hist[4]);
    assert!(close(hist[22], 6.1, 1e-4), "bin 220–229°: got {}, expected 6.1", hist[22]);
    assert!(close(hist[13], 1.0, 1e-4) && close(hist[30], 0.5, 1e-4), "bins 130° and 300°: got {} and {}", hist[13], hist[30]);
    let angles = dominant_orientations(&hist);
    assert!(angles.len() == 2 && close(angles[0], 45.0, 0.5) && close(angles[1], 225.0, 0.5), "A6: got {angles:?}, expected two keypoints, at 45° and 225°");
    // Add 30° to every direction: the orientations turn by 30°.
    let turned: Vec<(f32, f32)> = votes.iter().map(|&(m, d)| (m, d + 30.0)).collect();
    let angles = dominant_orientations(&orientation_histogram(&turned));
    assert!(angles.len() == 2 && close(angles[0], 75.0, 0.5) && close(angles[1], 255.0, 0.5), "A6 turned by 30°: got {angles:?}, expected 75° and 255°");
    // Directions outside [0, 360) wrap around.
    let hist = orientation_histogram(&[(1.0, -5.0), (1.0, 365.0)]);
    assert!(close(hist[35], 1.0, 1e-6) && close(hist[0], 1.0, 1e-6), "−5° belongs in bin 35, 365° in bin 0");
}
