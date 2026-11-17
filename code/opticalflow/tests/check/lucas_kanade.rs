// Checks for the coding exercises, using the numbers of the pen-and-paper exercises (B1 to B5)
// and frame pairs with a known motion.
// Run: cargo test -p opticalflow lucas_kanade::

use imgcourse::Image;
use opticalflow::lucas_kanade::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// A smooth textured image (a sum of waves), moved by (dx, dy): frame(x, y) = pattern(x − dx, y − dy).
fn pattern(size: usize, dx: f32, dy: f32) -> Image<f32> {
    Image::from_fn(size, size, |x, y| {
        let (x, y) = (x as f32 - dx, y as f32 - dy);
        0.5 + 0.15 * (x / 5.0).sin() + 0.15 * (y / 6.0 + 0.3 * x / 7.0).cos() + 0.1 * ((x + y) / 9.0).sin()
    })
}

/// The mean endpoint error inside a border of `m` pixels.
fn mean_epe(u: &Image<f32>, v: &Image<f32>, du: f32, dv: f32, m: usize) -> f32 {
    let (mut sum, mut n) = (0.0, 0.0);
    for y in m..u.height() - m {
        for x in m..u.width() - m {
            sum += (u[(x, y)] - du).hypot(v[(x, y)] - dv);
            n += 1.0;
        }
    }
    sum / n
}

#[test]
fn c1_gradients() {
    // A ramp of slope 0.1 in x, frame 2 is 0.05 brighter: Ix = 0.1, Iy = 0, It = 0.05.
    let f1 = Image::from_fn(6, 5, |x, _| 0.1 * x as f32);
    let f2 = f1.map(|v| v + 0.05);
    let (ix, iy, it) = gradients(&f1, &f2);
    assert!(close(ix[(2, 2)], 0.1, 1e-6) && close(iy[(2, 2)], 0.0, 1e-6) && close(it[(2, 2)], 0.05, 1e-6), "ramp: got Ix {}, Iy {}, It {}; expected 0.1, 0, 0.05", ix[(2, 2)], iy[(2, 2)], it[(2, 2)]);
    // The gradient is taken from the mean of the two frames.
    let g2 = Image::from_fn(6, 5, |x, _| 0.3 * x as f32);
    let (ix, _, _) = gradients(&f1, &g2);
    assert!(close(ix[(2, 2)], 0.2, 1e-6), "Ix of the mean of slopes 0.1 and 0.3: got {}, expected 0.2", ix[(2, 2)]);
}

#[test]
fn c2_lk_window_b1_b2_b4() {
    let (u, v) = lk_window(&[[1.0, 0.0, -2.0], [1.0, 0.0, -2.0], [0.0, 1.0, -1.0], [0.0, 1.0, -1.0]]).unwrap();
    assert!(close(u, 2.0, 1e-5) && close(v, 1.0, 1e-5), "B1: got ({u}, {v}), expected (2, 1)");
    assert!(lk_window(&[[1.0, 0.0, -2.0]; 4]).is_none(), "B2: all pixels on one edge, J is singular: expected None");
    let (u, v) = lk_window(&[[2.0, 0.0, -2.0], [0.0, 1.0, -1.0], [1.0, 1.0, -2.5]]).unwrap();
    assert!(close(u, 9.5 / 9.0, 1e-4) && close(v, 11.0 / 9.0, 1e-4), "B4: got ({u}, {v}), expected (1.06, 1.22)");
}

#[test]
fn c2_lk_solve_noise_b5() {
    let j = [10.0, 0.0, 0.1];
    let (u0, v0) = lk_solve(j, [-5.0, -0.2]).unwrap();
    let (u1, v1) = lk_solve(j, [-4.9, -0.1]).unwrap();
    assert!(close(u1 - u0, -0.01, 1e-4) && close(v1 - v0, -1.0, 1e-3), "B5: the estimate changes by ({}, {}), expected (−0.01, −1)", u1 - u0, v1 - v0);
}

#[test]
fn c3_min_eigenvalue_b2_b3() {
    assert!(close(min_eigenvalue([4.0, 0.0, 0.0]), 0.0, 1e-5), "B2: J = diag(4, 0): μ2 = 0");
    assert!(close(min_eigenvalue([4.0, 4.0, 4.0]), 0.0, 1e-5), "B3: J = [[4, 4], [4, 4]]: μ2 = 0");
    let m = min_eigenvalue([5.0, 1.0, 2.0]);
    assert!(close(m, 1.697, 1e-3), "B4: J = [[5, 1], [1, 2]]: got μ2 = {m}, expected 1.70");
}

#[test]
fn c4_lucas_kanade_small_shift() {
    let (f1, f2) = (pattern(48, 0.0, 0.0), pattern(48, 0.6, -0.4));
    let (u, v, mu2) = lucas_kanade(&f1, &f2, 2.0);
    let e = mean_epe(&u, &v, 0.6, -0.4, 6);
    assert!(e < 0.03, "a smooth pattern moved by (0.6, −0.4): mean endpoint error {e:.3} px, expected below 0.03");
    assert!(mu2[(24, 24)] > 0.0, "a textured window has μ2 > 0");
    // A flat frame pair: no gradients, no flow, μ2 = 0.
    let flat = Image::new(20, 20, 0.5);
    let (u, _, mu2) = lucas_kanade(&flat, &flat, 2.0);
    assert!(u[(10, 10)] == 0.0 && mu2[(10, 10)].abs() < 1e-9, "flat frames: flow 0 and μ2 0");
}

#[test]
fn c5_pyramid_rescues_large_motion() {
    let (f1, f2) = (pattern(64, 0.0, 0.0), pattern(64, 5.0, 3.0));
    let (u, v, _) = lucas_kanade(&f1, &f2, 3.0);
    let single = mean_epe(&u, &v, 5.0, 3.0, 12);
    let (u, v) = pyramid_lk(&f1, &f2, 3, 3, 3.0);
    let pyramid = mean_epe(&u, &v, 5.0, 3.0, 12);
    assert!(pyramid < 0.2, "moved by (5, 3), 3 levels × 3 iterations: mean endpoint error {pyramid:.2} px, expected below 0.2 (one solve: {single:.2})");
}
