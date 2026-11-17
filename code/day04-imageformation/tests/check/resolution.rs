// Checks for the coding exercises, using the numbers of the lecture and of the by-hand exercises.
// Run: cargo test -p day04 resolution::

use imgcourse::{samples, Image};
use day04::resolution::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// Positions −0.5 .. 0.5 µm every 1 nm; the middle sample is 0.
fn xs() -> Vec<f32> {
    (-500..=500).map(|i| i as f32 * 0.001).collect()
}

fn gaussian(sigma: f32) -> impl Fn(f32) -> f32 {
    move |x: f32| (-x * x / (2.0 * sigma * sigma)).exp()
}

#[test]
fn c1_profile_of_two_points() {
    let p = two_points(|x: f32| if x.abs() < 0.5 { 1.0 } else { 0.0 }, 2.0, &[-1.0, 0.0, 1.0, 3.0]);
    assert_eq!(p, vec![1.0, 0.0, 1.0, 0.0], "two box PSFs at −1 and +1, read at −1, 0, 1, 3");
    let g = two_points(gaussian(1.0), 2.0, &[0.0, 1.0]);
    let e = [2.0 * (-0.5f32).exp(), 1.0 + (-2.0f32).exp()];
    assert!(close(g[0], e[0], 1e-5) && close(g[1], e[1], 1e-5), "Gaussians σ = 1 at ±1: got {g:?}, expected {e:?}");
}

#[test]
fn c2_dip_simple() {
    let d = dip(&[1.0, 2.0, 1.0, 2.0, 1.0]);
    assert!(close(d, 0.5, 1e-6), "dip of (1, 2, 1, 2, 1): got {d}, expected 0.5");
    let d = dip(&[1.0, 2.0, 3.0, 2.0, 1.0]);
    assert!(close(d, 0.0, 1e-6), "one peak in the middle: got {d}, expected 0");
}

#[test]
fn c2_rayleigh_and_sparrow() {
    // NA 1.4, 520 nm: Rayleigh 0.61 λ/NA gives a dip of 26 %, Sparrow 0.47 λ/NA none.
    let h = |r: f32| airy(r, 1.4, 0.52);
    let rayleigh = dip(&two_points(h, 0.61 * 0.52 / 1.4, &xs()));
    assert!(close(rayleigh, 0.265, 0.01), "Rayleigh distance: dip {rayleigh}, expected about 0.26");
    let sparrow = dip(&two_points(h, 0.47 * 0.52 / 1.4, &xs()));
    assert!(sparrow < 0.005, "Sparrow distance: dip {sparrow}, expected about 0");
}

#[test]
fn c2_colour_lecture() {
    // Two points 227 nm apart at NA 1.4: in red light (650 nm) the dip almost vanishes,
    // in blue light (450 nm) it is about half the peak.
    let red = dip(&two_points(|r| airy(r, 1.4, 0.65), 0.227, &xs()));
    let blue = dip(&two_points(|r| airy(r, 1.4, 0.45), 0.227, &xs()));
    assert!(red < 0.02, "650 nm: dip {red}, expected below 2 %");
    assert!(close(blue, 0.51, 0.03), "450 nm: dip {blue}, expected about 0.51");
}

#[test]
fn c2_gaussian_sparrow_b9() {
    // Gaussian spots: one peak for d < 2σ, a dip for d > 2σ.
    let below = dip(&two_points(gaussian(0.1), 0.19, &xs()));
    let above = dip(&two_points(gaussian(0.1), 0.21, &xs()));
    assert!(close(below, 0.0, 1e-6), "σ = 0.1, d = 0.19: dip {below}, expected 0 (one peak)");
    assert!(above > 0.0, "σ = 0.1, d = 0.21: dip {above}, expected above 0");
}

#[test]
fn c3_fwhm() {
    let tri = fwhm(&[0.0, 1.0, 2.0, 1.0, 0.0], 1.0);
    assert!(close(tri, 2.0, 1e-5), "triangle (0, 1, 2, 1, 0): got {tri}, expected 2");
    let tri = fwhm(&[0.0, 0.0, 2.0, 0.0, 0.0], 0.5);
    assert!(close(tri, 0.5, 1e-5), "spike (0, 0, 2, 0, 0) with dx 0.5: got {tri}, expected 0.5");
    let xs = xs();
    let g: Vec<f32> = xs.iter().map(|&x| gaussian(0.1)(x - 0.0123)).collect();
    let w = fwhm(&g, 0.001);
    assert!(close(w, 0.2355, 0.001), "Gaussian σ = 0.1: got {w}, expected 2.355 σ = 0.2355");
    let a: Vec<f32> = xs.iter().map(|&x| airy(x, 1.4, 0.52)).collect();
    let w = fwhm(&a, 0.001);
    assert!(close(w, 0.191, 0.002), "Airy pattern, NA 1.4, 520 nm: got {w} µm, expected about 0.191 µm (0.51 λ/NA)");
}

fn blob(cx: f32, cy: f32, sigma: f32, background: f32) -> Image<f32> {
    Image::from_fn(25, 16, |x, y| {
        let (dx, dy) = (x as f32 - cx, y as f32 - cy);
        background + (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp()
    })
}

#[test]
fn c4_centroid() {
    let (x, y) = centroid(&blob(12.3, 7.6, 1.2, 0.1), 5, 0.1);
    assert!(close(x, 12.3, 0.02) && close(y, 7.6, 0.02), "bead at (12.3, 7.6), background subtracted: got ({x}, {y})");
    let (x, y) = centroid(&blob(12.3, 7.6, 1.2, 0.1), 5, 0.0);
    assert!(
        (x - 12.3).abs() > 0.05 || (y - 7.6).abs() > 0.05,
        "background not subtracted: got ({x}, {y}); the background should pull it towards the middle of the window, (12, 8)"
    );
}

#[test]
fn c5_average_of_clean_beads() {
    // Four beads at different sub-pixel positions, no noise: the average is the PSF.
    let sigma = 1.5;
    let centres = [(10.3, 10.7), (30.0, 12.25), (12.6, 30.1), (31.45, 29.5)];
    let img = Image::from_fn(42, 42, |x, y| {
        0.1 + centres.iter().map(|&(cx, cy)| (-((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)) / (2.0 * sigma * sigma)).exp()).sum::<f32>()
    });
    let psf = average_beads(&img, &centres, 16, 0.25, 0.1);
    assert_eq!((psf.width(), psf.height()), (33, 33), "size: got {} × {}, expected 33 × 33", psf.width(), psf.height());
    assert!(close(psf.sum(), 1.0, 1e-4), "the average sums to {}, expected 1", psf.sum());
    assert!(psf[(16, 16)] == psf.max(), "the peak should be in the middle, (16, 16)");
    let w = fwhm(psf.row(16), 0.25);
    assert!(close(w, 2.355 * sigma, 0.25), "FWHM of the average: got {w} pixels, expected about {} (bilinear sampling widens it a little)", 2.355 * sigma);
}

#[test]
fn c5_noise_falls_with_more_beads_b8() {
    let sigma = 1.5;
    let centres: Vec<(f32, f32)> = (0..16).map(|i| (8.0 + 16.0 * (i % 4) as f32 + 0.1 * i as f32, 8.0 + 16.0 * (i / 4) as f32 + 0.05 * i as f32)).collect();
    let clean = Image::from_fn(72, 72, |x, y| {
        centres.iter().map(|&(cx, cy)| (-((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)) / (2.0 * sigma * sigma)).exp()).sum::<f32>()
    });
    let noisy = samples::add_noise(&clean, 0.1, 7);
    let rms = |n: usize| {
        let a = average_beads(&noisy, &centres[..n], 6, 0.5, 0.0);
        let b = average_beads(&clean, &centres[..n], 6, 0.5, 0.0);
        let s: f32 = a.pixels().iter().zip(b.pixels()).map(|(p, q)| (p - q).powi(2)).sum();
        (s / a.pixels().len() as f32).sqrt()
    };
    let ratio = rms(16) / rms(1);
    assert!((0.12..0.45).contains(&ratio), "16 beads against 1: the noise fell to {ratio:.2} of one bead's, expected about 1/√16 = 0.25");
}
