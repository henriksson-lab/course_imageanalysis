// Path traces the closed box of the lecture: the noise against the number of samples, direct
// light only, and a tiny lamp with and without lamp sampling. Pictures in out/pathtrace/.
//   cargo run -p pathtrace                      (your code)
//   cargo run -p pathtrace --features solution  (the model solution)

use imgcourse::*;
use pathtrace::*;
use std::time::Instant;

// Pictures at 128 × 96; the error measurements at 48 × 36 (to keep the run short).
const W: usize = 128;
const H: usize = 96;
const W_SMALL: usize = 48;
const H_SMALL: usize = 36;
const EYE: Vec3 = v3(0.0, 1.0, 2.9);

/// For display: clip to [0, 1] and apply the sRGB-like gamma 1/2.2 (the path tracer works in
/// linear light).
fn display(img: &Image<[f32; 3]>) -> Image<[f32; 3]> {
    img.map(|c| c.map(|v| v.clamp(0.0, 1.0).powf(1.0 / 2.2)))
}

/// The root mean square difference between two images, over all pixels and colours, after
/// clipping to [0, 1] as for display (so that the lamp itself does not dominate).
fn rms(a: &Image<[f32; 3]>, b: &Image<[f32; 3]>) -> f32 {
    let mut sum = 0.0;
    for (x, y) in a.coords() {
        for k in 0..3 {
            sum += (a[(x, y)][k].clamp(0.0, 1.0) - b[(x, y)][k].clamp(0.0, 1.0)).powi(2);
        }
    }
    (sum / (3 * a.width() * a.height()) as f32).sqrt()
}

fn main() {
    let dir = "out/pathtrace";
    let start = Instant::now();
    let scene = cornell_box(0.25, 3.0);
    let settings = |spp: usize, seed: u64| Settings { spp, max_bounces: 4, light_sampling: true, seed };

    // The pictures: 1, 16 and 64 samples per pixel.
    for spp in [1, 16, 64] {
        save_rgb(&display(&render(&scene, EYE, W, H, settings(spp, spp as u64))), format!("{dir}/spp{spp}.png"));
    }
    // Direct light only: one bounce (eye → surface → lamp).
    let direct = render(&scene, EYE, W, H, Settings { max_bounces: 1, ..settings(32, 5) });
    save_rgb(&display(&direct), format!("{dir}/direct-only.png"));

    // The error for 1, 4, 16 and 64 samples per pixel, against a 1024-sample reference.
    let reference = render(&scene, EYE, W_SMALL, H_SMALL, settings(1024, 99));
    println!("RMS error against a 1024-sample reference (lamp sampling, up to 4 bounces, {W_SMALL} × {H_SMALL}):");
    let mut previous = f32::NAN;
    for spp in [1, 4, 16, 64] {
        let img = render(&scene, EYE, W_SMALL, H_SMALL, settings(spp, spp as u64));
        let e = rms(&img, &reference);
        println!("  {spp:4} spp: RMS {e:.4}   (previous ÷ this: {:.2})", previous / e);
        previous = e;
    }

    // A tiny lamp (0.1 × 0.1) with the same power: cosine sampling only, and lamp sampling.
    let tiny = cornell_box(0.05, 3.0);
    let tiny_ref = render(&tiny, EYE, W_SMALL, H_SMALL, settings(256, 98));
    for light_sampling in [false, true] {
        let s = Settings { light_sampling, ..settings(32, 7) };
        let small = render(&tiny, EYE, W_SMALL, H_SMALL, s);
        let img = render(&tiny, EYE, W, H, s);
        let name = if light_sampling { "tiny-lamp-sampled" } else { "tiny-cosine-only" };
        // Fireflies: pixels of the walls and floor (not the lamp itself) far brighter than white.
        let fireflies = img.pixels().iter().filter(|c| c[0] > 2.0 && c[0] < 100.0).count();
        println!("tiny lamp, 32 spp, {name}: RMS {:.3}, {fireflies} pixels brighter than 2", rms(&small, &tiny_ref));
        save_rgb(&display(&img), format!("{dir}/{name}.png"));
    }
    println!("{:.1} s; wrote the pictures to {dir}/", start.elapsed().as_secs_f32());
}
