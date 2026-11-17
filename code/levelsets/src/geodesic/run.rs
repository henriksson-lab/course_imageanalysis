// Segments a synthetic image with your edge-stopping function and geodesic active contour steps;
// pictures in out/geodesic/.
//   cargo run -p levelsets -- geodesic                      (your code)
//   cargo run -p levelsets --features solution -- geodesic  (the model solution)

use crate::geodesic::*;
use crate::common::{draw_contour, pieces};
use imgcourse::*;

const SIZE: usize = 128;

/// The test image (grey levels from 0 to 1) and its true object mask: three disks and a C, the
/// lower-left disk fading out towards its right side (a weak edge).
fn test_image() -> (Image<f32>, Image<bool>) {
    let (bg, high) = (60.0 / 255.0, 130.0 / 255.0);
    let mut img = Image::new(SIZE, SIZE, bg);
    let mut truth = Image::new(SIZE, SIZE, false);
    for (x, y) in img.coords() {
        let (fx, fy) = (x as f32, y as f32);
        let r = |cx: f32, cy: f32| ((fx - cx).powi(2) + (fy - cy).powi(2)).sqrt();
        let c_shape = (12.0..24.0).contains(&r(90.0, 90.0)) && !(fx > 90.0 && (fy - 90.0).abs() < 6.0);
        let contrast = if r(34.0, 34.0) < 18.0 || r(92.0, 36.0) < 15.0 || c_shape {
            high
        } else if r(36.0, 92.0) < 18.0 {
            // fades from full contrast on the left to 25 grey levels on the right
            high + (25.0 / 255.0 - high) * ((fx - 18.0) / 36.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        if contrast > 0.0 {
            img[(x, y)] = bg + contrast;
            truth[(x, y)] = true;
        }
    }
    (samples::add_noise(&img, 12.0 / 255.0, 5), truth)
}

/// Signed distance of a box (half-size `h`) around the centre of the image: the outside-in start.
fn box_start(h: f32) -> Image<f32> {
    Image::from_fn(SIZE, SIZE, |x, y| {
        let (dx, dy) = ((x as f32 - 63.5).abs() - h, (y as f32 - 63.5).abs() - h);
        (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt() + dx.max(dy).min(0.0)
    })
}

fn wrong(phi: &Image<f32>, truth: &Image<bool>) -> usize {
    phi.coords().filter(|&(x, y)| (phi[(x, y)] < 0.0) != truth[(x, y)]).count()
}

/// Run the geodesic active contour: braking balloon plus advection. (These runs are short
/// enough to do without reinitialisation.)
fn run(phi0: &Image<f32>, g: &Image<f32>, c: f32, eps: f32, a: f32, dt: f32, steps: usize) -> Image<f32> {
    let mut phi = phi0.clone();
    for _ in 0..steps {
        phi = braking_step(&phi, g, c, eps, dt);
        if a != 0.0 {
            phi = advection_step(&phi, g, a, dt);
        }
    }
    phi
}

pub fn main() {
    let dir = "out/geodesic";
    let (img, truth) = test_image();
    let k = 8.0 / 255.0;
    let g = edge_stopping(&gaussian_blur(&img, 1.5), k);
    save_gray(&g, format!("{dir}/g.png"));
    let mut sorted = g.pixels().to_vec();
    sorted.sort_by(f32::total_cmp);
    println!("g with sigma = 1.5: median {:.2}, smallest {:.3}", sorted[sorted.len() / 2], sorted[0]);

    // Outside-in: a box near the border shrinks (c = −0.4), with and without advection.
    let start = box_start(60.0);
    let (dt, steps) = (0.25, 720);
    let mut pics = Vec::new();
    for a in [5.0, 0.0] {
        let phi = run(&start, &g, -0.4, 0.3, a, dt, steps);
        println!("box, c = -0.4, eps = 0.3, a = {a}: t = {:.0}: {} piece(s), {} wrong pixels", dt * steps as f32, pieces(&phi, 0.0), wrong(&phi, &truth));
        let mut pic = gray_to_rgb(&img);
        draw_contour(&mut pic, &phi, [1.0, 0.9, 0.0]);
        pics.push(pic);
    }

    // Inside-out from a small seed in the disk with the weak edge: a strong and a gentle balloon.
    let seed = Image::from_fn(SIZE, SIZE, |x, y| ((x as f32 - 32.0).powi(2) + (y as f32 - 92.0).powi(2)).sqrt() - 10.0);
    let mut pic = gray_to_rgb(&img);
    for (c, a, colour) in [(1.0, 1.0, [1.0, 0.3, 0.2]), (0.3, 3.0, [0.2, 1.0, 0.3])] {
        let phi = run(&seed, &g, c, 0.3, a, 0.25, 240);
        let inside = phi.pixels().iter().filter(|&&v| v < 0.0).count();
        println!("seed in the weak disk (true size {} pixels), c = {c}, a = {a}, t = 60: {inside} pixels inside", truth.coords().filter(|&(x, y)| truth[(x, y)] && x < 60 && y > 64).count());
        draw_contour(&mut pic, &phi, colour);
    }
    pics.push(pic);
    let w = SIZE + 4;
    let sheet = Image::from_fn(2 * (3 * w - 4), 2 * SIZE, |x, y| {
        let (x, y) = (x / 2, y / 2);
        if x % w < SIZE { pics[x / w][(x % w, y)] } else { [1.0; 3] }
    });
    save_rgb(&sheet, format!("{dir}/segment.png"));
    println!("wrote the pictures to {dir}/");
}
