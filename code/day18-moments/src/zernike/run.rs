// Zernike moments of a letter F: rotated, mirrored and rebuilt; and the accuracy of the
// factorial formula. Results in out/zernike/.
//   cargo run -p day18 -- zernike                      (your code)
//   cargo run -p day18 --features solution -- zernike  (the model solution)

use imgcourse::*;
use crate::zernike::*;

/// The letter F of `samples::letter_f`, turned by `angle` degrees counter-clockwise about the
/// image centre and drawn with partial pixels (4 × 4 samples per pixel), `size × size`.
fn letter_f_turned(size: usize, angle: f64) -> Image<f32> {
    let (s, c) = angle.to_radians().sin_cos();
    let h = size as f64 / 2.0;
    Image::from_fn(size, size, |x, y| {
        let mut hits = 0;
        for i in 0..4 {
            for j in 0..4 {
                // the sample, relative to the centre with y up, turned back by the angle
                let (px, py) = (x as f64 + (i as f64 + 0.5) / 4.0 - h, h - (y as f64 + (j as f64 + 0.5) / 4.0));
                let (u, v) = ((c * px + s * py + h) / size as f64, (h - (-s * px + c * py)) / size as f64);
                let stem = (0.25..0.40).contains(&u) && (0.15..0.85).contains(&v);
                let top = (0.25..0.75).contains(&u) && (0.15..0.30).contains(&v);
                let middle = (0.25..0.62).contains(&u) && (0.45..0.58).contains(&v);
                if stem || top || middle {
                    hits += 1;
                }
            }
        }
        hits as f32 / 16.0
    })
}

/// All |Z_nm| with n <= 8, m >= 0.
fn magnitudes(img: &Image<f32>) -> Vec<f64> {
    let mut v = Vec::new();
    for n in 0..=8 {
        for m in (n % 2..=n).step_by(2) {
            v.push(zernike_moment(img, n, m as i32).abs() as f64);
        }
    }
    v
}

pub fn main() {
    let dir = "out/zernike";
    let size = 64;
    let f = letter_f_turned(size, 0.0);
    let mags = magnitudes(&f);
    let phase = |img: &Image<f32>, n: usize, m: i32| (zernike_moment(img, n, m).arg() as f64).to_degrees();
    println!("letter F, {size} × {size}: largest change of |Z_nm| (n <= 8), as a percentage of |Z00|,");
    println!("and the phases of Z11, Z22, Z31 (degrees):");
    println!("{:>22} {:>8} {:>8} {:>8} {:>8}", "", "|Z| diff", "Z11", "Z22", "Z31");
    let mirror = Image::from_fn(size, size, |x, y| f[(size - 1 - x, y)]);
    let cases = [("original", f.clone()), ("turned 30°", letter_f_turned(size, 30.0)), ("turned 90°", letter_f_turned(size, 90.0)), ("turned 180°", letter_f_turned(size, 180.0)), ("mirrored left–right", mirror)];
    let mut strip = Image::new(cases.len() * (size + 2), size, 0.5f32);
    for (k, (name, img)) in cases.iter().enumerate() {
        let m2 = magnitudes(img);
        let worst = mags.iter().zip(&m2).map(|(a, b)| (a - b).abs() / mags[0]).fold(0.0, f64::max);
        println!("{name:>22} {:7.2}% {:8.1} {:8.1} {:8.1}", 100.0 * worst, phase(img, 1, 1), phase(img, 2, 2), phase(img, 3, 1));
        for (x, y) in img.coords() {
            strip[(k * (size + 2) + x, y)] = img[(x, y)];
        }
    }
    save_gray(&strip, format!("{dir}/rotations.png"));

    // rebuild the F from its moments up to order N, on the disc
    let orders = [4, 10, 20, 30];
    let mut sheet = Image::new((orders.len() + 1) * (size + 2), size, 0.5f32);
    for (x, y) in f.coords() {
        sheet[(x, y)] = f[(x, y)];
    }
    let mut errors = Vec::new();
    for (k, &n) in orders.iter().enumerate() {
        let rec = reconstruct(&f, n);
        let (mut num, mut den) = (0.0, 0.0);
        for (x, y) in f.coords() {
            if polar(x, y, size).0 <= 1.0 {
                num += ((rec[(x, y)] - f[(x, y)]) as f64).powi(2);
                den += (f[(x, y)] as f64).powi(2);
            }
            sheet[((k + 1) * (size + 2) + x, y)] = rec[(x, y)];
        }
        errors.push(format!("N = {n}: {:.3}", (num / den).sqrt()));
    }
    println!();
    println!("relative error of the reconstruction inside the disc: {}", errors.join(", "));
    let big = Image::from_fn(sheet.width() * 2, sheet.height() * 2, |x, y| sheet[(x / 2, y / 2)]);
    save_gray(&big, format!("{dir}/reconstructions.png"));

    // the factorial formula against the recurrence
    println!();
    println!("largest difference on [0, 1] between the factorial formula and the recurrence for R_n0:");
    for n in [10, 20, 30, 40, 50] {
        let worst = (0..=1000).map(|i| i as f64 / 1000.0).map(|r| (radial(n, 0, r) - radial0_recurrence(n, r)).abs()).fold(0.0, f64::max);
        println!("  n = {n}: {worst:.1e}");
    }
    println!("wrote {dir}/rotations.png and {dir}/reconstructions.png (input, N = 4, 10, 20, 30)");
}
