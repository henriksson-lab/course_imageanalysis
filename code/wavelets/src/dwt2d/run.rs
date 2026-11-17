// The wavelet pyramid of the micrograph, keeping the largest coefficients in wavelet and Fourier
// bases, and shift variance; pictures in out/dwt2d/.
//   cargo run -p wavelets -- dwt2d                      (your code)
//   cargo run -p wavelets --features solution -- dwt2d  (the model solution)

use crate::dwt2d::*;
use imgcourse::*;

/// The pyramid for display: LL stretched to black..white, every detail band scaled on its own
/// (0 = mid-grey; ±3 times the band's RMS = black or white, larger values clipped).
fn pyramid_picture(c: &Image<f32>, levels: usize) -> Image<f32> {
    let mut out = Image::new(c.width(), c.height(), 0.5);
    let band = |out: &mut Image<f32>, x0: usize, y0: usize, w: usize, h: usize, ll: bool| {
        let part = c.crop(x0, y0, w, h);
        let (lo, hi) = (part.min(), part.max());
        let rms = (part.pixels().iter().map(|v| v * v).sum::<f32>() / part.pixels().len() as f32).sqrt();
        let m = (3.0 * rms).max(1e-12);
        for (x, y) in part.coords() {
            let v = part[(x, y)];
            out[(x0 + x, y0 + y)] = if ll { (v - lo) / (hi - lo) } else { (0.5 + 0.5 * v / m).clamp(0.0, 1.0) };
        }
    };
    let (mut w, mut h) = (c.width(), c.height());
    for _ in 0..levels {
        w /= 2;
        h /= 2;
        band(&mut out, w, 0, w, h, false); // V
        band(&mut out, 0, h, w, h, false); // H
        band(&mut out, w, h, w, h, false); // D
    }
    band(&mut out, 0, 0, w, h, true);
    out
}

/// Keep the given fraction of the (complex) Fourier coefficients and transform back.
fn fourier_keep(img: &Image<f32>, fraction: f32) -> Image<f32> {
    let spectrum = fft::fft2_real(img);
    let mut sizes: Vec<f32> = spectrum.pixels().iter().map(|z| z.abs()).collect();
    sizes.sort_by(|a, b| b.total_cmp(a));
    let threshold = sizes[(fraction * sizes.len() as f32).round() as usize - 1];
    let kept = spectrum.map(|z| if z.abs() >= threshold { z } else { Complex::new(0.0, 0.0) });
    fft::fft2(&kept, true).map(|z| z.re)
}

pub fn main() {
    let dir = "out/dwt2d";
    let cell = samples::cell();
    let levels = 4;

    // The pyramid and how the energy is spread.
    let c = dwt2d(&cell, &DB4, levels);
    // Shown twice as large, so that the small coarse bands can be seen.
    let pic = pyramid_picture(&c, levels);
    save_gray(&Image::from_fn(512, 512, |x, y| pic[(x / 2, y / 2)]), format!("{dir}/pyramid.png"));
    let total: f32 = c.pixels().iter().map(|v| v * v).sum();
    let ll = c.crop(0, 0, 256 >> levels, 256 >> levels);
    let ll_energy: f32 = ll.pixels().iter().map(|v| v * v).sum();
    println!("db4, {levels} levels: the {}×{} LL band ({:.1} % of the coefficients) holds {:.1} % of the energy", ll.width(), ll.height(), 100.0 / 256.0, 100.0 * ll_energy / total);

    // Keep the largest 1, 5, 20 % of the coefficients (grey levels 0..255 for the PSNR).
    let img = cell.map(|v| 255.0 * v);
    println!("\nPSNR (dB) keeping the largest coefficients, {levels} levels:");
    println!("  kept    Haar    db2    db4   Fourier");
    for fraction in [0.01, 0.05, 0.20] {
        let mut line = format!("  {:3.0} % ", 100.0 * fraction);
        for (name, h) in [("haar", &HAAR[..]), ("db2", &DB2[..]), ("db4", &DB4[..])] {
            let (kept, _) = keep_fraction(&dwt2d(&img, h, levels), fraction);
            let back = idwt2d(&kept, h, levels);
            line += &format!("  {:5.1}", psnr(&img, &back, 255.0));
            if fraction == 0.05 && name != "db2" {
                save_gray(&back.map(|v| v / 255.0), format!("{dir}/keep5-{name}.png"));
            }
        }
        let back = fourier_keep(&img, fraction);
        line += &format!("   {:5.1}", psnr(&img, &back, 255.0));
        if fraction == 0.05 {
            save_gray(&back.map(|v| v / 255.0), format!("{dir}/keep5-fourier.png"));
        }
        println!("{line}");
    }

    // Shift variance: move the image one pixel to the right (wrapping around) and compare the
    // detail coefficients.
    println!("\nshift by one pixel: relative change of the detail coefficients");
    let square = Image::from_fn(128, 128, |x, y| if (37..90).contains(&x) && (37..90).contains(&y) { 1.0 } else { 0.0 });
    for (name, image, h, lv) in [("square, Haar, 2 levels", &square, &HAAR[..], 2), ("micrograph, db4, 3 levels", &cell, &DB4[..], 3)] {
        let w = image.width();
        let shifted = Image::from_fn(w, image.height(), |x, y| image[((x + w - 1) % w, y)]);
        let (a, b) = (dwt2d(image, h, lv), dwt2d(&shifted, h, lv));
        let (mut diff, mut norm) = (0.0, 0.0);
        let llw = w >> lv;
        for (x, y) in a.coords() {
            if x >= llw || y >= llw {
                diff += (a[(x, y)] - b[(x, y)]).powi(2);
                norm += a[(x, y)].powi(2);
            }
        }
        // Energy of the level-1 V band before and after.
        let half = w / 2;
        let energy = |c: &Image<f32>| -> f32 { c.crop(half, 0, half, half).pixels().iter().map(|v| v * v).sum() };
        println!("  {name:26} {:5.1} %   (level-1 V energy {:.4} → {:.4})", 100.0 * (diff / norm).sqrt(), energy(&a), energy(&b));
    }
    println!("wrote pyramid.png and keep5-*.png to {dir}/");
}
