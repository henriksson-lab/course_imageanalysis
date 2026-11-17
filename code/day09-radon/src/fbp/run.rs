// Filtered back-projection of the Shepp–Logan phantom: plain against ramp, ramp against Hann
// with noise, too few angles, and a missing wedge. Pictures go to out/fbp/.
//   cargo run -p day09 -- fbp                      (your code)
//   cargo run -p day09 --features solution -- fbp  (the model solution)

use crate::common::*;
use crate::fbp::*;
use imgcourse::*;

fn rmse(a: &Image<f32>, b: &Image<f32>) -> f32 {
    let s: f32 = a.pixels().iter().zip(b.pixels()).map(|(p, q)| (p - q).powi(2)).sum();
    (s / a.pixels().len() as f32).sqrt()
}

/// The standard deviation in a flat 12 × 12 patch of the phantom's brain (value 0.2).
fn flat_noise(img: &Image<f32>) -> f32 {
    let patch = img.crop(80, 90, 12, 12);
    let m = patch.mean();
    (patch.pixels().iter().map(|v| (v - m).powi(2)).sum::<f32>() / patch.pixels().len() as f32).sqrt()
}

/// For display: values 0 to 0.5 as black to white (the phantom's soft tissue is 0.2).
fn window(img: &Image<f32>) -> Image<f32> {
    img.map(|v| v / 0.5)
}

pub fn main() {
    let dir = "out/fbp";
    let n = 128;
    let phantom = shepp_logan(n);
    save_gray(&window(&phantom), format!("{dir}/phantom.png"));
    let (ramp, hann) = (ramp_kernel(n), hann_kernel(n));

    // 1. A point from 16 angles: plain back-projection (a star) against FBP (stripes cancel).
    let c = centre(n) as usize;
    let mut point = Image::new(n, n, 0.0f32);
    point[(c + 10, c - 6)] = 1.0;
    let sixteen = angles(16);
    let sino = sinogram(&point, &sixteen);
    save_gray_stretched(&backproject(&sino, &sixteen, n), format!("{dir}/point16-plain.png"));
    save_gray_signed(&fbp(&sino, &sixteen, n, &ramp), format!("{dir}/point16-ramp.png"));

    // 2. 200 angles (the angle rule for 128 bins): plain, ramp and Hann.
    let thetas = angles(200);
    let sino = sinogram(&phantom, &thetas);
    let plain = backproject(&sino, &thetas, n);
    save_gray_stretched(&plain, format!("{dir}/plain.png"));
    for (name, h) in [("ramp", &ramp), ("hann", &hann)] {
        let r = fbp(&sino, &thetas, n, h);
        println!("200 angles, {name:<4}: RMSE {:.3}, noise in a flat patch {:.4}", rmse(&r, &phantom), flat_noise(&r));
        save_gray(&window(&r), format!("{dir}/{name}.png"));
    }

    // 3. Noise: Gaussian noise added to every line integral.
    let mut rng = Rng::new(3);
    let noisy = Image::from_fn(sino.width(), sino.height(), |a, k| sino[(a, k)] + 1.0 * rng.normal());
    for (name, h) in [("ramp", &ramp), ("hann", &hann)] {
        let r = fbp(&noisy, &thetas, n, h);
        println!("noisy data, {name:<4}: RMSE {:.3}, noise in a flat patch {:.4}", rmse(&r, &phantom), flat_noise(&r));
        save_gray(&window(&r), format!("{dir}/noisy-{name}.png"));
    }

    // 4. Too few angles: streaks.
    let few = angles(18);
    let r = fbp(&sinogram(&phantom, &few), &few, n, &hann);
    println!("18 angles, hann: RMSE {:.3}", rmse(&r, &phantom));
    save_gray(&window(&r), format!("{dir}/18-angles.png"));

    // 5. The missing wedge: tilts from −60° to +60° in 2° steps (the beam is vertical at 0°
    // tilt, so the rays are vertical: θ = 0 in the lecture's convention), a disc of diameter 61.
    let tilts = angles_between(-60.0, 60.0, 61);
    let cf = centre(n);
    let disc = Image::from_fn(n, n, |x, y| if (x as f32 - cf).powi(2) + (y as f32 - cf).powi(2) <= 30.5 * 30.5 { 1.0 } else { 0.0 });
    let r = fbp(&sinogram(&disc, &tilts), &tilts, n, &hann);
    save_gray(&disc, format!("{dir}/disc.png"));
    save_gray(&r.map(|v| v / 1.2), format!("{dir}/disc-wedge.png"));
    // width and height of the region above half the disc's value, through the centre
    let width = (0..n).filter(|&x| r[(x, c)] > 0.5).count();
    let height = (0..n).filter(|&y| r[(c, y)] > 0.5).count();
    println!("±60° tilts: the disc (diameter 61) comes out {width} wide and {height} tall");
    println!("wrote pictures to {dir}/");
}
