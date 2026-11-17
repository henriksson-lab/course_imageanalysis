// Runs your forward model on the course micrograph, treated as the true object with 40 nm
// pixels, and on stripe patterns (gratings). Results in out/psf/.
//   cargo run -p imageformation -- psf                      (your code)
//   cargo run -p imageformation --features solution -- psf  (the model solution)

use imgcourse::*;
use crate::psf::*;
use std::f32::consts::PI;

const PIXEL: f32 = 0.040; // µm per pixel in the object
const LAMBDA: f32 = 0.520; // µm, green light

/// The Gaussian approximation of an ideal lens: σ = 0.21 λ / NA, here in pixels.
fn sigma_pixels(na: f32) -> f32 {
    0.21 * LAMBDA / na / PIXEL
}

/// A grating 0.5 + 0.5 cos(2π x / period), period in pixels.
fn grating(period: f32) -> Image<f32> {
    Image::from_fn(256, 64, |x, _| 0.5 + 0.5 * (2.0 * PI * x as f32 / period).cos())
}

/// The contrast of the image of a grating relative to the object's, with its sign: negative
/// when bright and dark have swapped. Measured on row 32 over whole periods from x = 64,
/// away from the edges.
fn signed_contrast(image: &Image<f32>, period: f32) -> f32 {
    let length = ((128.0 / period).floor() * period).round() as usize;
    let xs = 64..64 + length;
    let n = xs.len() as f32;
    let mean = xs.clone().map(|x| image[(x, 32)]).sum::<f32>() / n;
    let (mut num, mut den) = (0.0, 0.0);
    for x in xs {
        let c = (2.0 * PI * x as f32 / period).cos();
        num += (image[(x, 32)] - mean) * c;
        den += 0.5 * c * c;
    }
    num / den
}

/// The PSF summed over its rows: the 1D profile that acts on a grating along x.
fn profile_x(psf: &Image<f32>) -> Vec<f32> {
    (0..psf.width()).map(|x| (0..psf.height()).map(|y| psf[(x, y)]).sum()).collect()
}

pub fn main() {
    let dir = "out/psf";
    let object = samples::cell();
    save_gray(&object, format!("{dir}/object.png"));

    // The micrograph through an oil objective (NA 1.4) and a dry one (NA 0.5).
    for na in [1.4, 0.5] {
        let psf = gaussian_psf(sigma_pixels(na));
        println!("NA {na}: Gaussian PSF σ = {:.2} pixels = {:.0} nm, {} × {} pixels", sigma_pixels(na), sigma_pixels(na) * PIXEL * 1000.0, psf.width(), psf.height());
        save_gray(&blur(&object, &psf, Border::Mirror), format!("{dir}/image-na{}.png", (na * 10.0) as u32));
    }
    // 1 µm of motion along x: 25 pixels.
    let motion = motion_psf(25);
    save_gray(&blur(&object, &motion, Border::Mirror), format!("{dir}/image-motion.png"));

    // Gratings through the NA 1.4 lens (Gaussian approximation), compared with the curve of
    // an ideal lens.
    let psf = gaussian_psf(sigma_pixels(1.4));
    println!("\ngratings, NA 1.4, 520 nm (cut-off {:.2} cycles/µm):", 2.0 * 1.4 / LAMBDA);
    println!("  period    cycles/µm   contrast in the image   OTF of the PSF   ideal lens");
    for period_nm in [1000.0, 500.0, 320.0, 250.0, 160.0] {
        let period = period_nm / 1000.0 / PIXEL;
        let k = 1.0 / (period * PIXEL);
        let image = blur(&grating(period), &psf, Border::Clamp);
        println!(
            "  {period_nm:5} nm   {k:8.2}   {:20.3}   {:14.3}   {:10.3}",
            signed_contrast(&image, period),
            otf(&profile_x(&psf), 1.0 / period),
            lens_mtf(k, 1.4, LAMBDA)
        );
    }

    // Gratings through 1 µm of motion along x.
    println!("\ngratings, 1 µm of motion along x:");
    println!("  period    contrast in the image   OTF");
    for period_nm in [2000.0, 1000.0, 700.0, 500.0] {
        let period = period_nm / 1000.0 / PIXEL;
        let image = blur(&grating(period), &motion, Border::Clamp);
        println!("  {period_nm:5} nm   {:20.3}   {:6.3}", signed_contrast(&image, period), otf(&profile_x(&motion), 1.0 / period));
        if period_nm == 700.0 {
            save_gray(&grating(period), format!("{dir}/grating-700nm.png"));
            save_gray(&image, format!("{dir}/grating-700nm-motion.png"));
        }
    }
    println!("\nwrote the pictures to {dir}/");
}
