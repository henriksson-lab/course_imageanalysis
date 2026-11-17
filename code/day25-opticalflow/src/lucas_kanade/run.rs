// Lucas–Kanade on the micrograph moved by a known shift: one solve, warping, the pyramid,
// the window size under noise, and the confidence μ2; pictures in out/lucas-kanade/.
//   cargo run -p day25 -- lucas-kanade                      (your code)
//   cargo run -p day25 --features solution -- lucas-kanade  (the model solution)

use imgcourse::*;
use crate::lucas_kanade::*;

/// Frame 1 (the micrograph, lightly blurred) and frame 2 moved by `(dx, dy)`, with noise.
fn frames(dx: f32, dy: f32, noise: f32) -> (Image<f32>, Image<f32>) {
    let f1 = gaussian_blur(&samples::cell(), 1.0);
    let f2 = f1.map_xy(|x, y, _| bilinear(&f1, x as f32 - dx, y as f32 - dy));
    (samples::add_noise(&f1, noise, 1), samples::add_noise(&f2, noise, 2))
}

/// The mean endpoint error, 16 px border excluded (where the shift brings in unknown pixels).
fn mean_epe(u: &Image<f32>, v: &Image<f32>, dx: f32, dy: f32) -> f32 {
    let (mut sum, mut n) = (0.0, 0.0);
    for y in 16..u.height() - 16 {
        for x in 16..u.width() - 16 {
            sum += (u[(x, y)] - dx).hypot(v[(x, y)] - dy);
            n += 1.0;
        }
    }
    sum / n
}

pub fn main() {
    let dir = "out/lucas-kanade";
    let noise = 0.01;
    println!("mean endpoint error (px), micrograph moved by s · (0.8, 0.6), noise {noise}, ρ = 3:");
    println!("  shift s   one solve   3 warping iterations   pyramid 4 levels × 3");
    for s in [1.0, 2.0, 4.0, 8.0] {
        let (dx, dy) = (0.8 * s, 0.6 * s);
        let (f1, f2) = frames(dx, dy, noise);
        let (u, v, _) = lucas_kanade(&f1, &f2, 3.0);
        let one = mean_epe(&u, &v, dx, dy);
        let (u, v) = pyramid_lk(&f1, &f2, 1, 3, 3.0);
        let warped = mean_epe(&u, &v, dx, dy);
        let (u, v) = pyramid_lk(&f1, &f2, 4, 3, 3.0);
        let pyramid = mean_epe(&u, &v, dx, dy);
        println!("  {s:5.0}    {one:8.2}   {warped:13.2}          {pyramid:13.2}");
        if s == 8.0 {
            save_rgb(&u.zip_map(&v, |a, b| flow_colour(a, b, 10.0)), format!("{dir}/flow-8px-pyramid.png"));
            let (u, v, _) = lucas_kanade(&f1, &f2, 3.0);
            save_rgb(&u.zip_map(&v, |a, b| flow_colour(a, b, 10.0)), format!("{dir}/flow-8px-one-solve.png"));
        }
    }

    // Window size under strong noise, shift 1 px.
    print!("noise 0.05, shift (0.8, 0.6), one solve:");
    let (f1, f2) = frames(0.8, 0.6, 0.05);
    for rho in [1.5, 3.0, 6.0] {
        let (u, v, _) = lucas_kanade(&f1, &f2, rho);
        print!("  ρ = {rho}: {:.2} px", mean_epe(&u, &v, 0.8, 0.6));
    }
    println!();

    // Confidence: μ2, and the error where μ2 is large and where it is small (noise 0.05).
    let (u, v, mu2) = lucas_kanade(&f1, &f2, 3.0);
    let mut sorted: Vec<f32> = mu2.pixels().to_vec();
    sorted.sort_by(f32::total_cmp);
    let threshold = sorted[sorted.len() / 10]; // the lowest tenth of μ2
    let (mut hi, mut nhi, mut lo, mut nlo) = (0.0, 0.0, 0.0, 0.0);
    for y in 16..240 {
        for x in 16..240 {
            let e = (u[(x, y)] - 0.8).hypot(v[(x, y)] - 0.6);
            if mu2[(x, y)] >= threshold { (hi, nhi) = (hi + e, nhi + 1.0) } else { (lo, nlo) = (lo + e, nlo + 1.0) }
        }
    }
    println!("noise 0.05, ρ = 3: mean endpoint error {:.3} px where μ2 is above its lowest tenth, {:.3} px in the lowest tenth", hi / nhi, lo / nlo);
    save_gray_stretched(&mu2.map(|m| m.max(0.0).sqrt()), format!("{dir}/confidence.png"));
    save_rgb(&u.zip_map(&v, |a, b| flow_colour(a, b, 2.0)), format!("{dir}/flow-1px.png"));
    let masked = Image::from_fn(256, 256, |x, y| if mu2[(x, y)] >= threshold { flow_colour(u[(x, y)], v[(x, y)], 2.0) } else { [0.5, 0.5, 0.5] });
    save_rgb(&masked, format!("{dir}/flow-1px-confident.png"));
    save_gray(&f1, format!("{dir}/frame1.png"));
    println!("wrote the pictures to {dir}/ (colour wheel: hue = direction, red = right; white = still)");
}

/// The colour wheel: hue = direction (red = right, counter-clockwise on screen), saturation =
/// speed / `max`, white = still.
fn flow_colour(u: f32, v: f32, max: f32) -> [f32; 3] {
    let angle = (-v).atan2(u).to_degrees().rem_euclid(360.0);
    let s = (u.hypot(v) / max).min(1.0);
    let h = angle / 60.0;
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    let (r, g, b) = match h as usize {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
    };
    [1.0 - s + s * r, 1.0 - s + s * g, 1.0 - s + s * b]
}
