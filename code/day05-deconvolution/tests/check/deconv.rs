// Checks for the coding exercises, using the numbers of the by-hand exercises.
// Run: cargo test -p day05 deconv::

use day05::deconv::*;
use imgcourse::{samples, Image};

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_wiener_b1() {
    for (h, e) in [(1.0, 0.990), (0.5, 1.923), (0.1, 5.0), (0.05, 4.0), (0.01, 0.990)] {
        let g = wiener_gain(h, 100.0);
        assert!(close(g, e, 2e-3), "Wiener, SNR 100, H = {h}: got {g}, expected {e}");
    }
}

#[test]
fn c1_truncated_b2() {
    for (h, e) in [(1.0, 1.0), (0.5, 2.0), (0.2, 5.0), (0.1, 0.0), (0.05, 0.0), (-0.5, -2.0)] {
        let g = truncated_gain(h, 0.1);
        assert!(close(g, e, 1e-5), "truncated, ε = 0.1, H = {h}: got {g}, expected {e}");
    }
}

#[test]
fn c1_tikhonov_two_pixels_b3() {
    // Gains of the sum part (s = 1) and the difference part (s = 0.5), energy penalty.
    for (lambda, sum, diff) in [(0.0, 1.0, 2.0), (0.1, 0.909, 1.429), (1.0, 0.5, 0.4)] {
        let (a, b) = (tikhonov_gain(1.0, lambda, 1.0), tikhonov_gain(0.5, lambda, 1.0));
        assert!(close(a, sum, 1e-3) && close(b, diff, 1e-3), "λ = {lambda}: gains ({a}, {b}), expected ({sum}, {diff})");
    }
}

#[test]
fn c1_wiener_b5() {
    for (h, e) in [(0.454, 2.10), (0.0425, 3.60), (0.0072, 0.72)] {
        let g = wiener_gain(h, 100.0);
        assert!(close(g, e, 0.01), "Wiener, SNR 100, H = {h}: got {g}, expected {e}");
    }
}

#[test]
fn c2_gaussian_otf_b5() {
    let otf = gaussian_otf(2.0, 40, 20);
    assert_eq!((otf.width(), otf.height()), (40, 20), "size: got {} × {}", otf.width(), otf.height());
    // f = 4/40, 8/40, 10/40 = 0.1, 0.2, 0.25 cycles/pixel along x
    for (x, e) in [(0, 1.0), (4, 0.454), (8, 0.0425), (10, 0.0072)] {
        assert!(close(otf[(x, 0)], e, 1e-3), "σ = 2, pixel ({x}, 0) = {} cycles/pixel: got {}, expected {e}", x as f32 / 40.0, otf[(x, 0)]);
    }
    // negative frequencies: x = 36 is −4/40
    assert!(close(otf[(36, 0)], otf[(4, 0)], 1e-6), "pixel (36, 0) is −0.1 cycles/pixel: got {}, expected {}", otf[(36, 0)], otf[(4, 0)]);
    assert!(close(otf[(0, 2)], otf[(4, 0)], 1e-6), "pixel (0, 2) is 2/20 = 0.1 cycles/pixel along y: got {}, expected {}", otf[(0, 2)], otf[(4, 0)]);
}

#[test]
fn c2_laplacian_symbol_b9() {
    for (fx, fy, e) in [(0.0, 0.0, 0.0), (0.5, 0.0, 4.0), (0.0, 0.5, 4.0), (0.5, 0.5, 8.0), (0.25, 0.0, 2.0)] {
        let l = laplacian_symbol(fx, fy);
        assert!(close(l, e, 1e-5), "L({fx}, {fy}): got {l}, expected {e}");
    }
}

#[test]
fn c3_apply_gain() {
    let g = samples::cell().crop(40, 40, 32, 16);
    let same = apply_gain(&g, &Image::new(32, 16, 1.0));
    for (x, y) in g.coords() {
        assert!(close(same[(x, y)], g[(x, y)], 1e-4), "gain 1 everywhere, pixel ({x}, {y}): got {}, expected {}", same[(x, y)], g[(x, y)]);
    }
    // Only frequency 0 kept: every pixel becomes the mean.
    let mut only_mean = Image::new(32, 16, 0.0);
    only_mean[(0, 0)] = 1.0;
    let flat = apply_gain(&g, &only_mean);
    assert!(close(flat[(5, 7)], g.mean(), 1e-4), "gain 1 at frequency 0 only: got {}, expected the mean {}", flat[(5, 7)], g.mean());
}

/// The micrograph crop, blurred with σ = 2 (periodic), no noise.
fn blurred() -> (Image<f32>, Image<f32>, Image<f32>) {
    let f = samples::cell().crop(64, 64, 64, 64);
    let otf = gaussian_otf(2.0, 64, 64);
    let g = apply_gain(&f, &otf);
    (f, g, otf)
}

#[test]
fn c4_means_b6() {
    let (_, g, otf) = blurred();
    let energy = restore(&g, &otf, Method::TikhonovEnergy(0.1));
    let rough = restore(&g, &otf, Method::TikhonovRoughness(0.1));
    assert!(close(energy.mean(), g.mean() / 1.1, 1e-4), "energy penalty, λ = 0.1: mean {}, expected {} / 1.1 = {}", energy.mean(), g.mean(), g.mean() / 1.1);
    assert!(close(rough.mean(), g.mean(), 1e-4), "roughness penalty: mean {}, expected {} (kept)", rough.mean(), g.mean());
}

#[test]
fn c4_tikhonov_energy_is_wiener() {
    let (_, g, otf) = blurred();
    let a = restore(&g, &otf, Method::TikhonovEnergy(1.0 / 45.0));
    let b = restore(&g, &otf, Method::Wiener(45.0));
    for (x, y) in g.coords() {
        assert!(close(a[(x, y)], b[(x, y)], 1e-4), "pixel ({x}, {y}): Tikhonov λ = 1/45 gives {}, Wiener SNR 45 gives {}", a[(x, y)], b[(x, y)]);
    }
}

#[test]
fn c4_inverse_without_noise() {
    let f = samples::cell().crop(64, 64, 32, 32);
    let otf = gaussian_otf(0.7, 32, 32);
    let back = restore(&apply_gain(&f, &otf), &otf, Method::Inverse);
    for (x, y) in f.coords() {
        assert!(close(back[(x, y)], f[(x, y)], 2e-3), "inverse, no noise, pixel ({x}, {y}): restored {}, object {}", back[(x, y)], f[(x, y)]);
    }
}
