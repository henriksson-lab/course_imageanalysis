// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day20 spot_filters::

use imgcourse::{samples, Image};
use day20::spot_filters::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// The 5 × 5 patch of B1: row r0 is y = 0, column c0 is x = 0.
fn b1_patch() -> Image<f32> {
    Image::from_rows(&[
        [1., 2., 2., 1., 0.],
        [2., 7., 3., 1., 1.],
        [1., 3., 2., 4., 2.],
        [0., 2., 5., 6., 3.],
        [1., 1., 2., 3., 2.],
    ])
}

#[test]
fn c1_local_maxima_b1() {
    let m1 = local_maxima(&b1_patch(), 1);
    assert_eq!(m1, vec![(1, 1), (3, 3)], "B1(a), r = 1: expected r1 c1 (7) and r3 c3 (6) as (x, y) = (1, 1), (3, 3)");
    let m2 = local_maxima(&b1_patch(), 2);
    assert_eq!(m2, vec![(1, 1)], "B1(b), r = 2: expected only r1 c1 (7)");
}

#[test]
fn c1_plateau_b1c() {
    // ..., 3, 5, 5, 2, ...: the first 5 is the candidate, the second has an equal neighbour before it.
    let row = Image::from_rows(&[[3., 5., 5., 2.]]);
    assert_eq!(local_maxima(&row, 1), vec![(1, 0)], "B1(c): in 3, 5, 5, 2 only the first 5 should be kept");
}

#[test]
fn c2_kernel_sums_and_noise_b8() {
    let g = spot_kernel(1.5, false);
    assert_eq!((g.width(), g.height()), (11, 11), "radius ceil(3 · 1.5) = 5, size 11");
    assert!(close(g.sum(), 1.0, 1e-4), "the Gaussian kernel should sum to 1, got {}", g.sum());
    assert!(g[(5, 5)] > g[(4, 5)] && close(g[(4, 5)], g[(6, 5)], 1e-6), "largest in the centre, symmetric");
    // B8: sum of squared weights about 1/(4 π σ²) = 0.0354, so the noise sd drops to 0.188 σn.
    let ss: f32 = g.pixels().iter().map(|w| w * w).sum();
    assert!(close(ss, 0.0354, 0.0005), "sum of squared weights: got {ss}, expected about 0.0354 (B8)");
    let z = spot_kernel(1.5, true);
    assert!(close(z.sum(), 0.0, 1e-4), "the zero-mean kernel should sum to 0, got {}", z.sum());
    assert!(z[(5, 5)] > 0.0 && z[(0, 0)] < 0.0, "zero-mean kernel: positive centre, negative surround");
}

#[test]
fn c3_background_b5() {
    // B5(a), (b): a constant background of 100.
    let flat = Image::new(21, 21, 100.0f32);
    let g = correlate(&flat, &spot_kernel(1.5, false));
    let z = correlate(&flat, &spot_kernel(1.5, true));
    assert!(close(g[(10, 10)], 100.0, 1e-2), "Gaussian on a constant 100: got {}, expected 100", g[(10, 10)]);
    assert!(close(z[(10, 10)], 0.0, 1e-3), "zero-mean on a constant 100: got {}, expected 0", z[(10, 10)]);
    // B5(c): a ramp of 0.5 per pixel; the symmetric zero-mean filter gives 0 away from the border.
    let ramp = Image::from_fn(21, 21, |x, _| 0.5 * x as f32);
    let z = correlate(&ramp, &spot_kernel(1.5, true));
    assert!(close(z[(10, 10)], 0.0, 1e-3), "zero-mean on a ramp: got {}, expected 0", z[(10, 10)]);
}

#[test]
fn c3_matched_filter_snr_b2() {
    // B2: peak 30, noise sd 12, σ = 1: SNR 2.5 for one pixel, 2.5 √π = 4.43 after the matched filter.
    let spot = Image::from_fn(15, 15, |x, y| {
        let d2 = (x as f32 - 7.0).powi(2) + (y as f32 - 7.0).powi(2);
        30.0 * (-d2 / 2.0).exp()
    });
    let k = spot_kernel(1.0, false);
    let signal = correlate(&spot, &k)[(7, 7)];
    let noise = 12.0 * k.pixels().iter().map(|w| w * w).sum::<f32>().sqrt();
    let snr = signal / noise;
    assert!(close(snr, 4.43, 0.03), "B2(b): matched-filter SNR got {snr}, expected 4.43");
}

#[test]
fn c4_atrous_b3() {
    let spike = Image::from_rows(&[[0., 0., 0., 0., 16., 0., 0., 0., 0.]]);
    let (w, a2) = atrous(&spike, 2);
    // a1 = spike - w1
    let a1: Vec<f32> = (0..9).map(|x| spike[(x, 0)] - w[0][(x, 0)]).collect();
    let expected = [0., 0., 1., 4., 6., 4., 1., 0., 0.];
    for x in 0..9 {
        assert!(close(a1[x], expected[x], 1e-4), "B3(a): a1 = {a1:?}, expected {expected:?}");
    }
    assert!(close(w[0][(4, 0)], 10.0, 1e-4), "B3(a): w1 at the centre got {}, expected 10", w[0][(4, 0)]);
    assert!(close(a2[(4, 0)], 2.75, 1e-4), "B3(c): a2 at the centre got {}, expected 44/16 = 2.75", a2[(4, 0)]);
    let energy = |img: &Image<f32>| img.pixels().iter().map(|v| v * v).sum::<f32>();
    assert!(close(energy(&w[0]), 134.0, 1e-2), "B3(d): energy of w1 got {}, expected 134", energy(&w[0]));
    assert!(energy(&w[1]) < 30.0, "B3(d): energy of w2 got {}, expected about 21", energy(&w[1]));
}

#[test]
fn c4_atrous_telescopes() {
    let img = samples::cell().crop(60, 80, 40, 30);
    let (w, last) = atrous(&img, 4);
    assert_eq!(w.len(), 4, "four levels w1..w4");
    for (x, y) in img.coords() {
        let sum = last[(x, y)] + w.iter().map(|wj| wj[(x, y)]).sum::<f32>();
        assert!(close(sum, img[(x, y)], 1e-4), "pixel ({x}, {y}): a4 + w1 + ... + w4 = {sum}, but the image is {}", img[(x, y)]);
    }
}

#[test]
fn c5_product_keeps_the_spot() {
    let noise = samples::add_noise(&Image::new(64, 64, 0.2), 0.04, 3);
    let img = Image::from_fn(64, 64, |x, y| {
        let d2 = (x as f32 - 30.0).powi(2) + (y as f32 - 34.0).powi(2);
        noise[(x, y)] + 0.4 * (-d2 / (2.0 * 1.5 * 1.5)).exp()
    });
    let (w, _) = atrous(&img, 3);
    let p = multiscale_product(&w, &[2, 3], 3.0);
    assert!(p[(30, 34)] > 0.0, "the product should be positive at the spot (30, 34), got {}", p[(30, 34)]);
    let nonzero = p.pixels().iter().filter(|&&v| v > 0.0).count();
    assert!(nonzero < 60, "in pure noise the product should be 0 almost everywhere; {nonzero} of 4096 pixels are positive");
}
