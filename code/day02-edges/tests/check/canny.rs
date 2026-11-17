// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day02 canny::

use day02::canny::*;
use imgcourse::Image;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

/// The noisy step of B1 as an image one row high.
fn b1_signal() -> Image<f32> {
    Image::from_rows(&[[10., 8., 13., 15., 9., 12., 8., 30., 27., 33., 29., 31., 26., 30.]])
}

#[test]
fn c1_gaussian_kernel() {
    let g = gaussian_kernel(2.0);
    assert_eq!(g.len(), 17, "σ = 2, cut at 4σ: expected 17 taps, got {}", g.len());
    let sum: f32 = g.iter().sum();
    assert!(close(sum, 1.0, 1e-5), "the weights should sum to 1, got {sum}");
    assert!(close(g[0], g[16], 1e-7) && g[8] > g[7], "the kernel should be symmetric with its peak in the middle: {g:?}");
    assert!(close(g[8], 0.1995, 2e-3), "σ = 2: the centre weight should be about 1/(√(2π)·2) = 0.1995, got {}", g[8]);
}

#[test]
fn c1_derivative_kernel_ramp_and_noise() {
    for sigma in [1.0f32, 2.0] {
        let w = gaussian_derivative_kernel(sigma);
        let r = (w.len() / 2) as f32;
        let on_ramp: f32 = w.iter().enumerate().map(|(i, v)| (i as f32 - r) * v).sum();
        let on_flat: f32 = w.iter().sum();
        assert!(close(on_ramp, 1.0, 1e-4), "σ = {sigma}: a ramp of slope 1 should give 1, got {on_ramp}");
        assert!(close(on_flat, 0.0, 1e-5), "σ = {sigma}: a flat image should give 0, got {on_flat}");
        assert!(w[w.len() - 2] > 0.0, "σ = {sigma}: the weights right of the centre should be positive (no flip)");
        // Lecture, "how much noise survives?": the sum of squared weights is about 1/(4√π σ³).
        let noise: f32 = w.iter().map(|v| v * v).sum();
        let expected = 1.0 / (4.0 * std::f32::consts::PI.sqrt() * sigma.powi(3));
        assert!(close(noise, expected, 0.05 * expected), "σ = {sigma}: noise variance factor {noise}, expected about {expected}");
    }
}

#[test]
fn c2_filter_rows_b1() {
    // B1(a): the central difference ½[−1 0 1], x = 1 … 8.
    let diff = filter_rows(&b1_signal(), &[-0.5, 0.0, 0.5]);
    let expected = [1.5, 3.5, -2.0, -1.5, -0.5, 9.0, 9.5, 1.5];
    for (i, e) in expected.iter().enumerate() {
        assert!(close(diff[(i + 1, 0)], *e, 1e-4), "B1(a), x = {}: got {}, expected {e}", i + 1, diff[(i + 1, 0)]);
    }
    // B1(b): the 5-tap Gaussian, x = 2 … 8.
    let smooth = filter_rows(&b1_signal(), &[0.05, 0.25, 0.4, 0.25, 0.05]);
    let expected = [11.9, 12.5, 11.4, 11.3, 15.5, 23.0, 28.4];
    for (i, e) in expected.iter().enumerate() {
        assert!(close(smooth[(i + 2, 0)], *e, 0.051), "B1(b), x = {}: got {}, expected {e}", i + 2, smooth[(i + 2, 0)]);
    }
    // B1(d): the 7-tap kernel does both at once; at x = 7 it gives 6.45.
    let both = filter_rows(&b1_signal(), &[-0.025, -0.125, -0.175, 0.0, 0.175, 0.125, 0.025]);
    assert!(close(both[(7, 0)], 6.45, 1e-3), "B1(d), x = 7: got {}, expected 6.45", both[(7, 0)]);
}

#[test]
fn c2_filter_columns_matches_rows() {
    // Filtering the columns of an image is filtering the rows of its transpose.
    let img = imgcourse::samples::cell().crop(60, 80, 12, 9);
    let transposed = Image::from_fn(img.height(), img.width(), |x, y| img[(y, x)]);
    let w = [0.1, -0.3, 0.5, 0.2, 0.7];
    let a = filter_columns(&img, &w);
    let b = filter_rows(&transposed, &w);
    for (x, y) in img.coords() {
        assert!(close(a[(x, y)], b[(y, x)], 1e-5), "pixel ({x}, {y}): columns {} but rows of the transpose {}", a[(x, y)], b[(y, x)]);
    }
}

#[test]
fn c2_gradient_of_a_ramp_and_a_step() {
    // A plane that rises 2.5 per pixel to the right and 1 per pixel going up (y down: −1).
    let plane = Image::from_fn(40, 40, |x, y| 2.5 * x as f32 - y as f32);
    let (lx, ly) = gradient(&plane, 2.0);
    assert!(close(lx[(20, 20)], 2.5, 1e-3), "Lx in the middle of the plane: got {}, expected 2.5", lx[(20, 20)]);
    assert!(close(ly[(20, 20)], -1.0, 1e-3), "Ly in the middle of the plane: got {}, expected −1", ly[(20, 20)]);
    // Lecture demo notes: a step of 120 at σ = 2 gives a bump of 120/(2√(2π)) ≈ 24.
    let step = imgcourse::samples::step(40, 10, 50.0, 170.0); // the step is between x = 19 and 20
    let (lx, _) = gradient(&step, 2.0);
    let peak = lx[(19, 5)].max(lx[(20, 5)]);
    assert!(close(peak, 23.9, 0.6), "step of 120, σ = 2: peak {peak}, expected about 24");
}

#[test]
fn c3_direction_sector_b2() {
    // B2: (a) 53° → 45°, (b) 180° → 0°, (c) −45° → 135°, (d) −90° → 90°.
    let cases = [((3.0, -4.0), 1), ((-2.0, 0.0), 0), ((1.0, 1.0), 3), ((0.0, 5.0), 2)];
    for ((lx, ly), sector) in cases {
        let got = direction_sector(lx, ly);
        assert_eq!(got, sector, "B2, Lx = {lx}, Ly = {ly}: got sector {got} ({}°), expected {sector} ({}°)", 45 * got, 45 * sector);
    }
}

#[test]
fn c3_nms_b3() {
    let mag = Image::from_rows(&[
        [2., 4., 9., 5., 1.],
        [1., 5., 10., 6., 2.],
        [2., 6., 8., 7., 3.],
        [1., 3., 9., 4., 2.],
        [0., 4., 8., 6., 1.],
    ]);
    // Directions of the inner 3 × 3 pixels (B3), as gradients: 0° = (1, 0), 45° = (1, −1), 90° = (0, −1).
    let mut lx = Image::new(5, 5, 1.0f32);
    let mut ly = Image::new(5, 5, 0.0f32);
    (lx[(2, 2)], ly[(2, 2)]) = (1.0, -1.0); // r2 c2: 45°
    (lx[(1, 3)], ly[(1, 3)]) = (0.0, -1.0); // r3 c1: 90°
    let out = non_max_suppression(&mag, &lx, &ly);
    for y in 1..4 {
        for x in 1..4 {
            let kept = out[(x, y)] > 0.0;
            let expected = x == 2; // only column c2 survives
            assert_eq!(kept, expected, "B3, row r{y} column c{x} (magnitude {}): kept = {kept}, expected {expected}", mag[(x, y)]);
        }
    }
}

#[test]
fn c4_hysteresis_b4() {
    let s = Image::from_rows(&[
        [0., 2., 9., 5., 1., 0.],
        [1., 3., 6., 0., 0., 2.],
        [0., 5., 1., 0., 3., 3.],
        [2., 4., 0., 0., 5., 0.],
        [6., 1., 0., 2., 6., 1.],
        [5., 0., 3., 7., 0., 0.],
    ]);
    let edge = hysteresis(&s, 4.0, 8.0);
    // B4(b): kept (row, column), from ring 0 to ring 5.
    let kept = [(0, 2), (0, 3), (1, 2), (2, 1), (3, 1), (4, 0), (5, 0)];
    for (r, c) in kept {
        assert!(edge[(c, r)], "B4: r{r} c{c} (value {}) should be kept", s[(c, r)]);
    }
    let n = edge.pixels().iter().filter(|&&e| e).count();
    assert_eq!(n, 7, "B4: expected 7 edge pixels, got {n} (the weak chain r3 c4, r4 c4, r5 c3 has no seed)");
}
