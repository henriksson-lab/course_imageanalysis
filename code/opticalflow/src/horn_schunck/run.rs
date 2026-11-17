// Horn–Schunck on a textured ring around a flat disc (filling-in, the warping residual) and on
// the micrograph (the choice of λ); pictures in out/horn-schunck/.
//   cargo run -p opticalflow -- horn-schunck                      (your code)
//   cargo run -p opticalflow --features solution -- horn-schunck  (the model solution)

use crate::common::{blur, flow_colour};
use crate::horn_schunck::*;
use imgcourse::*;

const SHIFT: (f32, f32) = (1.5, 0.0);

/// A 128 × 128 scene: a textured ring (radius 22 to 44) around a flat disc, on a flat
/// background, moved by `(dx, dy)`. The texture is smooth random noise.
fn ring(dx: f32, dy: f32) -> Image<f32> {
    let noise = samples::add_noise(&Image::new(160, 160, 0.0), 1.0, 5);
    let texture = blur(&noise, 2.0);
    let s = 0.25 / texture.max();
    let scene = Image::from_fn(160, 160, |x, y| {
        let r = (x as f32 - 80.0).hypot(y as f32 - 80.0);
        if (22.0..44.0).contains(&r) { 0.5 + s * texture[(x, y)] } else { 0.5 }
    });
    Image::from_fn(128, 128, |x, y| bilinear(&scene, x as f32 + 16.0 - dx, y as f32 + 16.0 - dy))
}

/// The mean endpoint error against a uniform true flow, inside a border of `m` pixels; and the
/// same over the disc of radius 15 at the centre of a 128 × 128 image.
fn epe(u: &Image<f32>, v: &Image<f32>, truth: (f32, f32), m: usize) -> (f32, f32) {
    let (mut all, mut n, mut centre, mut nc) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for y in m..u.height() - m {
        for x in m..u.width() - m {
            let e = endpoint_error((u[(x, y)], v[(x, y)]), truth);
            all += e;
            n += 1.0;
            if (x as f32 - 64.0).hypot(y as f32 - 64.0) < 15.0 {
                centre += e;
                nc += 1.0;
            }
        }
    }
    (all / n, centre / nc.max(1.0))
}

pub fn main() {
    let dir = "out/horn-schunck";
    // 1. Filling-in: the ring scene moved by 1.5 px.
    let (f1, f2) = (samples::add_noise(&ring(0.0, 0.0), 0.005, 1), samples::add_noise(&ring(SHIFT.0, SHIFT.1), 0.005, 2));
    save_gray(&f1, format!("{dir}/ring-frame1.png"));
    let (ix, iy, it) = gradients(&f1, &f2);
    let lambda = 0.001;
    println!("ring scene moved by 1.5 px, λ = {lambda}:");
    println!("  sweeps   u at the centre   mean EPE   EPE in the flat disc");
    let (mut u, mut v) = (Image::new(128, 128, 0.0), Image::new(128, 128, 0.0));
    let mut done = 0;
    for sweeps in [10, 100, 300, 1000, 2000] {
        while done < sweeps {
            (u, v) = hs_sweep(&ix, &iy, &it, &u, &v, lambda);
            done += 1;
        }
        let (all, centre) = epe(&u, &v, SHIFT, 4);
        println!("  {sweeps:6}   {:12.2}   {all:10.2}   {centre:10.2}", u[(64, 64)]);
        if sweeps == 100 || sweeps == 1000 {
            save_rgb(&u.zip_map(&v, |a, b| flow_colour(a, b, 2.0)), format!("{dir}/ring-flow-{sweeps}.png"));
        }
        if sweeps == 100 {
            // The warping residual: is it larger where the flow is wrong?
            let r = warp_residual(&f1, &f2, &u, &v);
            let (mut rc, mut nc, mut ra, mut na) = (0.0, 0.0, 0.0, 0.0);
            for (x, y) in r.coords() {
                if (4..124).contains(&x) && (4..124).contains(&y) {
                    ra += r[(x, y)].abs();
                    na += 1.0;
                    if (x as f32 - 64.0).hypot(y as f32 - 64.0) < 15.0 {
                        rc += r[(x, y)].abs();
                        nc += 1.0;
                    }
                }
            }
            println!("           after 100 sweeps: mean |residual| {:.4} in the flat disc, {:.4} overall", rc / nc, ra / na);
            save_gray_signed(&r, format!("{dir}/ring-residual-100.png"));
        }
    }

    // 2. The middle of the micrograph moved by (0.8, 0.6) with noise: the mean EPE against λ (300 sweeps).
    let cell = blur(&samples::cell(), 1.0).crop(64, 64, 128, 128);
    let moved = cell.map_xy(|x, y, _| bilinear(&cell, x as f32 - 0.8, y as f32 - 0.6));
    let (g1, g2) = (samples::add_noise(&cell, 0.02, 3), samples::add_noise(&moved, 0.02, 4));
    let (ix, iy, it) = gradients(&g1, &g2);
    println!("the middle 128 × 128 of the micrograph moved by (0.8, 0.6), noise 0.02, 300 sweeps:");
    for lambda in [0.00001, 0.0001, 0.001, 0.01, 0.1] {
        let (u, v) = horn_schunck(&ix, &iy, &it, lambda, 300);
        println!("  λ = {lambda:<7}: mean EPE {:.3} px", epe(&u, &v, (0.8, 0.6), 8).0);
        let name = match lambda {
            0.00001 => "small",
            0.01 => "medium",
            0.1 => "large",
            _ => continue,
        };
        save_rgb(&u.zip_map(&v, |a, b| flow_colour(a, b, 2.0)), format!("{dir}/cell-flow-lambda-{name}.png"));
    }
    println!("wrote the pictures to {dir}/ (colour wheel: hue = direction, red = right; white = still)");
}
