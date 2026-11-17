// Two points at the resolution limit, and a simulated bead slide from which your code
// measures the PSF. Results in out/resolution/.
//   cargo run -p day04 -- resolution                      (your code)
//   cargo run -p day04 --features solution -- resolution  (the model solution)

use imgcourse::*;
use crate::resolution::*;

const NA: f32 = 1.4;
const LAMBDA: f32 = 0.520; // µm
const PIXEL: f32 = 0.065; // µm in the sample: a 6.5 µm camera pixel behind a 100× objective

/// A Poisson-distributed photon count with the given mean (Knuth's method for small means,
/// the normal approximation for large ones).
fn poisson(rng: &mut Rng, mean: f32) -> f32 {
    if mean > 50.0 {
        return (mean + mean.sqrt() * rng.normal()).round().max(0.0);
    }
    let limit = (-mean).exp();
    let (mut k, mut p) = (0.0, rng.uniform());
    while p > limit {
        k += 1.0;
        p *= rng.uniform();
    }
    k
}

/// Enlarge an averaged PSF 4 times (nearest neighbour) and scale its peak to 1, 0 = black.
fn picture(psf: &Image<f32>) -> Image<f32> {
    let peak = psf.max();
    Image::from_fn(psf.width() * 4, psf.height() * 4, |x, y| psf[(x / 4, y / 4)] / peak)
}

pub fn main() {
    let dir = "out/resolution";
    let xs: Vec<f32> = (-500..=500).map(|i| i as f32 * 0.001).collect(); // µm, middle = 0

    // Two points.
    println!("two points, NA {NA}:");
    for (name, d) in [("Rayleigh 0.61 λ/NA", 0.61 * LAMBDA / NA), ("Sparrow 0.47 λ/NA", 0.47 * LAMBDA / NA)] {
        let p = two_points(|r| airy(r, NA, LAMBDA), d, &xs);
        println!("  {name:20} d = {:3.0} nm: dip {:5.1} %", d * 1000.0, 100.0 * dip(&p));
    }
    for lambda in [0.450, 0.520, 0.650] {
        let p = two_points(|r| airy(r, NA, lambda), 0.227, &xs);
        println!("  227 nm apart at {:3.0} nm:   dip {:5.1} %", lambda * 1000.0, 100.0 * dip(&p));
    }
    let single: Vec<f32> = xs.iter().map(|&x| airy(x, NA, LAMBDA)).collect();
    println!("  FWHM of the Airy pattern: {:.0} nm", 1000.0 * fwhm(&single, 0.001));

    // A bead slide: 30 small beads, 500 photons each, a background of 10 photons per pixel,
    // Poisson noise. The PSF is the Airy pattern sampled at the pixel centres.
    let (size, beads, photons, background) = (256, 30, 500.0, 10.0);
    let r = 20isize;
    let mut norm = 0.0;
    for dy in -r..=r {
        for dx in -r..=r {
            norm += airy(PIXEL * ((dx * dx + dy * dy) as f32).sqrt(), NA, LAMBDA);
        }
    }
    let mut rng = Rng::new(3);
    let mut centres = Vec::new();
    while centres.len() < beads {
        let c = (24.0 + 208.0 * rng.uniform(), 24.0 + 208.0 * rng.uniform());
        if centres.iter().all(|&(x, y): &(f32, f32)| (x - c.0).hypot(y - c.1) > 24.0) {
            centres.push(c);
        }
    }
    let mut clean = Image::new(size, size, 0.0f32);
    for &(cx, cy) in &centres {
        for dy in -r..=r {
            for dx in -r..=r {
                let (x, y) = (cx.round() as isize + dx, cy.round() as isize + dy);
                if clean.contains(x, y) {
                    let d = PIXEL * (x as f32 - cx).hypot(y as f32 - cy);
                    clean[(x as usize, y as usize)] += photons * airy(d, NA, LAMBDA) / norm;
                }
            }
        }
    }
    let slide = Image::from_fn(size, size, |x, y| poisson(&mut rng, clean[(x, y)] + background));
    println!("\nbead slide: {beads} beads, {photons} photons each, background {background} per pixel, {:.0} nm pixels", PIXEL * 1000.0);
    println!("  brightest pixel of a bead (no noise): {:.0} photons above the background", clean.max());
    save_gray_stretched(&slide, format!("{dir}/bead-slide.png"));

    // Find each bead's centre in a small window around where it is, then average.
    let find = |img: &Image<f32>, bg: f32| -> Vec<(f32, f32)> {
        centres
            .iter()
            .map(|&(cx, cy)| {
                let (x0, y0) = (cx.round() as usize - 6, cy.round() as usize - 6);
                let (x, y) = centroid(&img.crop(x0, y0, 13, 13), 3, bg);
                (x0 as f32 + x, y0 as f32 + y)
            })
            .collect()
    };
    let (n, step) = (24, 0.25); // 49 × 49 points, a quarter pixel apart
    let width = |psf: &Image<f32>| 1000.0 * fwhm(psf.row(n), step * PIXEL);
    let truth = average_beads(&clean, &centres, n, step, 0.0);
    println!("  FWHM of the true PSF on these pixels: {:.0} nm", width(&truth));
    let found = find(&slide, background);
    let error: f32 = found.iter().zip(&centres).map(|(a, b)| PIXEL * 1000.0 * (a.0 - b.0).hypot(a.1 - b.1)).sum::<f32>() / beads as f32;
    println!("  centres found with a mean error of {error:.0} nm");
    for k in [1, 2, 3, 5, 10, 20, 30] {
        let psf = average_beads(&slide, &found[..k], n, step, background);
        println!("  {k:2} beads averaged: FWHM {:.0} nm", width(&psf));
        if k == 1 || k == 30 {
            save_gray(&picture(&psf), format!("{dir}/psf-{k}-beads.png"));
        }
    }
    let forgot = average_beads(&slide, &find(&slide, 0.0), n, step, 0.0);
    println!("  30 beads, background not subtracted: FWHM {:.0} nm", width(&forgot));
    save_gray(&picture(&forgot), format!("{dir}/psf-30-beads-no-background.png"));
    println!("\nwrote the pictures to {dir}/");
}
