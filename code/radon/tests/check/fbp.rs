// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p radon fbp::

use radon::common::{angles, centre, sinogram};
use radon::fbp::*;
use imgcourse::Image;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_ram_lak_b2() {
    for (k, e) in [(0, 0.25), (1, -0.1013), (-1, -0.1013), (2, 0.0), (3, -0.01126), (-3, -0.01126), (5, -0.00405)] {
        assert!(close(ram_lak(k), e, 1e-4), "B2(a), h[{k}]: got {:.4}, expected {e}", ram_lak(k));
    }
    for (k_max, e) in [(1, 0.047), (3, 0.025)] {
        let sum: f32 = ramp_kernel(k_max).iter().sum();
        assert!(close(sum, e, 1e-3), "B2(b), sum over |k| ≤ {k_max}: got {sum:.3}, expected {e}");
    }
    let sum: f32 = ramp_kernel(2000).iter().sum();
    assert!(sum.abs() < 1e-3, "B2(b): the full kernel sums to 0, got {sum:.4} for |k| ≤ 2000");
    assert_eq!(ramp_kernel(3).len(), 7, "ramp_kernel(K) has 2K + 1 weights");
}

#[test]
fn c2_filter_box_b1() {
    let h = [-0.1, 0.25, -0.1];
    let out = filter(&[0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], &h);
    let e = [0.0, -0.1, 0.15, 0.05, 0.15, -0.1, 0.0];
    for i in 0..7 {
        assert!(close(out[i], e[i], 1e-5), "B1(a), bin {i}: got {:.3}, expected {}", out[i], e[i]);
    }
    let out = filter(&[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0], &h);
    let e = [0.0, -0.1, 0.15, 0.05, 0.05, 0.05, 0.15, -0.1, 0.0];
    for i in 0..9 {
        assert!(close(out[i], e[i], 1e-5), "B1(b), bin {i}: got {:.3}, expected {}", out[i], e[i]);
    }
    let out = filter(&[1.0, 2.0], &[0.0, 0.0, 1.0, 0.0, 0.0]);
    assert_eq!(out, vec![1.0, 2.0], "a kernel with only a 1 in the middle changes nothing");
}

#[test]
fn c3_fbp_of_a_disc() {
    let n = 65;
    let c = centre(n);
    let disc = Image::from_fn(n, n, |x, y| if (x as f32 - c).powi(2) + (y as f32 - c).powi(2) <= 15.0 * 15.0 { 1.0 } else { 0.0 });
    let thetas = angles(100);
    let r = fbp(&sinogram(&disc, &thetas), &thetas, n, &ramp_kernel(n));
    let inside = r.crop(27, 27, 11, 11).mean();
    assert!(close(inside, 1.0, 0.05), "the middle of a disc of value 1: got {inside:.3}, expected 1");
    let outside = (r[(5, 32)] + r[(32, 5)] + r[(59, 32)] + r[(32, 59)]) / 4.0;
    assert!(outside.abs() < 0.05, "outside the disc: got {outside:.3}, expected 0 (no glow)");
}

#[test]
fn c4_hann_kernel() {
    let h = hann_kernel(3);
    assert_eq!(h.len(), 7, "hann_kernel(K) has 2K + 1 weights");
    let e0 = 0.125 - 0.5 / std::f32::consts::PI.powi(2);
    assert!(close(h[3], e0, 1e-5), "h_Hann[0] = 1/8 − 1/(2π²) = {e0:.4}, got {:.4}", h[3]);
    // the same as filtering the ramp kernel with (1/4, 1/2, 1/4)
    let smoothed = filter(&ramp_kernel(4), &[0.25, 0.5, 0.25]);
    for k in 0..7 {
        assert!(close(h[k], smoothed[k + 1], 1e-6), "weight {k}: hann_kernel gives {:.5}, smoothing the ramp gives {:.5}", h[k], smoothed[k + 1]);
    }
    let sum: f32 = hann_kernel(2000).iter().sum();
    assert!(sum.abs() < 1e-3, "a tapered ramp still sums to 0, got {sum:.4}");
}
