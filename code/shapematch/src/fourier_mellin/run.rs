// Fourier–Mellin on the micrograph: view B is view A rotated, scaled and shifted; the program
// finds the angle, the scale and the shift, and shows where the method breaks (no taper with
// uneven light, a scale out of range, noise with and without the high-pass weight). Then the
// easy case: a known centre, one rotational correlation. Pictures in out/fourier-mellin/.
//   cargo run -p shapematch -- fourier-mellin                      (your code)
//   cargo run -p shapematch --features solution -- fourier-mellin  (the model solution)

use imgcourse::*;
use crate::fourier_mellin::*;

const N: usize = 128;
const DIR: &str = "out/fourier-mellin";

/// The views A and B: 128 × 128 around (128, 128) of the micrograph; B rotated by `angle`
/// degrees (clockwise), scaled by `scale` and shifted by `(tx, ty)`. Noise of SD `noise` (grey
/// levels of 255) on both; `shade` grey levels of uneven light, fixed to the camera.
fn views(angle: f32, scale: f32, tx: f32, ty: f32, noise: f32, shade: f32) -> (Image<f32>, Image<f32>) {
    let cell = samples::cell();
    let light = |img: Image<f32>| img.map_xy(|x, y, v| v + shade / 255.0 * (x + y) as f32 / (2 * N) as f32);
    let a = warp_similarity(&cell, N, N, 0.0, 1.0, 0.0, 0.0, 128.0, 128.0);
    let b = warp_similarity(&cell, N, N, angle, scale, tx, ty, 128.0, 128.0);
    (light(samples::add_noise(&a, noise / 255.0, 1)), light(samples::add_noise(&b, noise / 255.0, 2)))
}

/// Moves shift 0 to the middle of a correlation surface.
fn centred(c: &Image<f32>) -> Image<f32> {
    let (w, h) = (c.width(), c.height());
    Image::from_fn(w, h, |x, y| c[((x + w / 2) % w, (y + h / 2) % h)])
}

const BASE: Settings = Settings { taper: true, highpass: true, both: true, n_theta: 256, n_r: 128, r_min: 2.0, r_max: 62.0 };

#[allow(clippy::too_many_arguments)]
fn case(name: &str, angle: f32, scale: f32, tx: f32, ty: f32, noise: f32, shade: f32, s: Settings) {
    let (a, b) = views(angle, scale, tx, ty, noise, shade);
    let e = fourier_mellin(&a, &b, s);
    let ok = |d: bool| if d { "" } else { "  <- wrong" };
    let da = ((e.angle - angle).rem_euclid(360.0) + 180.0).rem_euclid(360.0) - 180.0;
    let good = da.abs() < 1.0 && (e.scale / scale - 1.0).abs() < 0.02 && (e.tx - tx).hypot(e.ty - ty) < 1.0;
    println!(
        "{name:20} true {angle:6.1}° × {scale:4.2} ({tx:4.0}, {ty:4.0})   found {:7.2}° × {:5.3} ({:6.2}, {:6.2})   peaks {:.3} / {:.3}{}",
        e.angle, e.scale, e.tx, e.ty, e.rot_peak, e.shift_peak, ok(good)
    );
    // pictures: the views, |F| (log), the log-polar grids, the surface, B with angle and scale undone
    let lp = |img: &Image<f32>| log_polar(&magnitude_spectrum(img, s.taper, s.highpass), s.r_min, s.r_max, s.n_r, s.n_theta);
    let (lp_a, lp_b) = (lp(&a), lp(&b));
    save_gray(&a, format!("{DIR}/{name}-a.png"));
    save_gray(&b, format!("{DIR}/{name}-b.png"));
    save_gray_stretched(&magnitude_spectrum(&b, s.taper, s.highpass).map(|v| (1.0 + v).ln()), format!("{DIR}/{name}-spectrum-b.png"));
    save_gray_stretched(&lp_a.map(|v| (1.0 + v).ln()), format!("{DIR}/{name}-logpolar-a.png"));
    save_gray_stretched(&lp_b.map(|v| (1.0 + v).ln()), format!("{DIR}/{name}-logpolar-b.png"));
    save_gray_signed(&centred(&phase_correlation(&lp_b, &lp_a, false)), format!("{DIR}/{name}-surface.png"));
    let back = warp_similarity(&b, N, N, -e.angle, 1.0 / e.scale, 0.0, 0.0, N as f32 / 2.0, N as f32 / 2.0);
    let overlay = Image::from_fn(N, N, |x, y| [a[(x, y)], back[(x, y)], a[(x, y)]]);
    save_rgb(&overlay, format!("{DIR}/{name}-overlay.png"));
}

pub fn main() {
    println!("Views of 128 × 128; log-polar grid 256 angles over 180°, 128 radii from 2 to 62 (log).");
    println!("Taper and high-pass on unless named; noise SD 5 grey levels unless named.\n");
    case("rotation-30", 30.0, 1.0, 0.0, 0.0, 5.0, 0.0, BASE);
    case("rotation-30-scale-1.5", 30.0, 1.5, 12.0, -8.0, 5.0, 0.0, BASE);
    case("rotation-200", 200.0, 1.0, 10.0, 5.0, 5.0, 0.0, BASE);
    case("rotation-200-one", 200.0, 1.0, 10.0, 5.0, 5.0, 0.0, Settings { both: false, ..BASE });
    case("shade-taper", 30.0, 1.0, 0.0, 0.0, 5.0, 400.0, BASE);
    case("shade-no-taper", 30.0, 1.0, 0.0, 0.0, 5.0, 400.0, Settings { taper: false, ..BASE });
    case("scale-1.8", 20.0, 1.8, 0.0, 0.0, 5.0, 0.0, BASE);
    case("scale-3", 20.0, 3.0, 0.0, 0.0, 5.0, 0.0, BASE);
    case("noise-25", 30.0, 1.0, 0.0, 0.0, 25.0, 0.0, BASE);
    case("noise-25-no-highpass", 30.0, 1.0, 0.0, 0.0, 25.0, 0.0, Settings { highpass: false, ..BASE });

    // A known centre: a disc of radius 48 around (112, 80), B turned by 50° about it.
    println!("\nKnown centre (112, 80), radius 48, 256 angles over 360°:");
    let cell = samples::cell();
    let size = 98;
    let disc_a = samples::add_noise(&warp_similarity(&cell, size, size, 0.0, 1.0, 0.0, 0.0, 112.0, 80.0), 10.0 / 255.0, 3);
    let disc_b = samples::add_noise(&warp_similarity(&cell, size, size, 50.0, 1.0, 0.0, 0.0, 112.0, 80.0), 10.0 / 255.0, 4);
    let c = size as f32 / 2.0;
    let (pa, pb) = (polar(&disc_a, c, c, 48.0, 48, 256), polar(&disc_b, c, c, 48.0, 48, 256));
    let weights: Vec<f32> = (0..48).map(|i| 48.0 * i as f32 / 47.0).collect();
    let score = rotational_correlation(&pa, &pb, &weights);
    let best = (0..256usize).max_by(|&i, &j| score[i].total_cmp(&score[j])).unwrap();
    let second = (0..256usize).filter(|&t| t.abs_diff(best) > 3 && t.abs_diff(best) < 253).map(|t| score[t]).fold(f32::MIN, f32::max);
    println!("  peak at column {best} = {:.1}°, second-highest value away from it {:.0} % of the peak", best as f32 * 360.0 / 256.0, 100.0 * second / score[best]);
    save_gray(&pa, format!("{DIR}/known-polar-a.png"));
    save_gray(&pb, format!("{DIR}/known-polar-b.png"));
    println!("wrote the pictures to {DIR}/");
}
