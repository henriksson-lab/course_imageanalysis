// Describes points of the micrograph and of a turned, zoomed, brighter copy, and counts how
// often the nearest descriptor is the true partner; pictures in out/descriptors/.
// The frames come from the known transform, so that only the descriptors are tested.
//   cargo run -p day24 -- descriptors                      (your code)
//   cargo run -p day24 --features solution -- descriptors  (the model solution)

use crate::common::blur;
use crate::descriptors::*;
use imgcourse::*;

const ANGLE: f32 = 30.0; // view 2 is view 1 turned by 30° counter-clockwise on the screen,
const ZOOM: f32 = 1.2; // zoomed 1.2 times about the centre,
const GAIN: f32 = 1.5; // 1.5 times brighter, with noise.
const SIGMA: f32 = 2.0; // the keypoint scale in view 1

/// A keypoint frame: x, y, σ, θ (degrees).
type Frame = (f32, f32, f32, f32);

/// Where a point of view 1 lands in view 2.
fn forward(x: f32, y: f32, c: f32) -> (f32, f32) {
    let (s, co) = ANGLE.to_radians().sin_cos();
    let (dx, dy) = (x - c, y - c);
    // Counter-clockwise on the screen (y down): (dx, dy) → (dx cos + dy sin, −dx sin + dy cos).
    (c + ZOOM * (co * dx + s * dy), c + ZOOM * (-s * dx + co * dy))
}

fn view2(img: &Image<f32>) -> Image<f32> {
    let c = img.width() as f32 / 2.0;
    let (s, co) = ANGLE.to_radians().sin_cos();
    let turned = Image::from_fn(img.width(), img.height(), |x, y| {
        let (dx, dy) = ((x as f32 - c) / ZOOM, (y as f32 - c) / ZOOM);
        bilinear(img, c + co * dx - s * dy, c + s * dx + co * dy)
    });
    samples::add_noise(&turned.map(|v| GAIN * v), 0.08, 7)
}

/// Points on a grid whose 15 × 15 neighbourhood has enough contrast, inside both views.
fn points(img: &Image<f32>) -> Vec<(f32, f32)> {
    let c = img.width() as f32 / 2.0;
    let mut pts = Vec::new();
    for y in (40..216).step_by(10) {
        for x in (40..216).step_by(10) {
            let (u, v) = forward(x as f32, y as f32, c);
            let inside = (30.0..226.0).contains(&u) && (30.0..226.0).contains(&v);
            let patch = img.crop(x - 7, y - 7, 15, 15);
            let mean = patch.mean();
            let sd = (patch.pixels().iter().map(|p| (p - mean) * (p - mean)).sum::<f32>() / 225.0).sqrt();
            if inside && sd > 0.08 {
                pts.push((x as f32, y as f32));
            }
        }
    }
    pts
}

/// The fraction of points whose nearest descriptor in view 2 is their true partner.
fn nearest_is_partner(d1: &[Vec<f32>], d2: &[Vec<f32>]) -> usize {
    (0..d1.len())
        .filter(|&i| {
            let nearest = (0..d2.len()).min_by(|&a, &b| distance(&d1[i], &d2[a]).total_cmp(&distance(&d1[i], &d2[b]))).unwrap();
            nearest == i
        })
        .count()
}

pub fn main() {
    let dir = "out/descriptors";
    let img = samples::cell();
    let img2 = view2(&img);
    let c = img.width() as f32 / 2.0;
    let pts = points(&img);
    let n = pts.len();
    // The frame of every point in both views: with θ (view 2 turned by 30°) or without (θ = 0).
    let frames = |with_theta: bool, error: f32| -> (Vec<Frame>, Vec<Frame>) {
        let f1 = pts.iter().map(|&(x, y)| (x, y, SIGMA, 0.0)).collect();
        let f2 = pts
            .iter()
            .map(|&(x, y)| {
                let (u, v) = forward(x, y, c);
                (u + error, v, ZOOM * SIGMA, if with_theta { ANGLE } else { 0.0 })
            })
            .collect();
        (f1, f2)
    };
    let patches = |im: &Image<f32>, f: &[Frame]| -> Vec<Image<f32>> { f.iter().map(|&(x, y, s, t)| sample_patch(im, x, y, s, t, 16)).collect() };
    let pct = |k: usize| 100.0 * k as f32 / n as f32;

    println!("{n} points; the nearest descriptor in view 2 is the true partner for:");
    for (name, with_theta, error) in [("with θ", true, 0.0), ("without θ", false, 0.0), ("with θ, frame 2 px off", true, 2.0), ("with θ, frame 4 px off", true, 4.0)] {
        let (f1, f2) = frames(with_theta, error);
        let (p1, p2) = (patches(&img, &f1), patches(&img2, &f2));
        let hist = nearest_is_partner(&p1.iter().map(gradient_descriptor).collect::<Vec<_>>(), &p2.iter().map(gradient_descriptor).collect::<Vec<_>>());
        let patch = nearest_is_partner(&p1.iter().map(normalised_patch).collect::<Vec<_>>(), &p2.iter().map(normalised_patch).collect::<Vec<_>>());
        println!("  {name:24} gradient histogram {hist:3} ({:3.0} %), normalised patch {patch:3} ({:3.0} %)", pct(hist), pct(patch));
    }

    // Binary descriptors: 256 random pairs in the 16 × 16 patch, on images smoothed with σ = 1.5.
    let mut rng = Rng::new(3);
    let pairs: Vec<_> = (0..256).map(|_| ((rng.below(16), rng.below(16)), (rng.below(16), rng.below(16)))).collect();
    let (s1, s2) = (blur(&img, 1.5), blur(&img2, 1.5));
    let (f1, f2) = frames(true, 0.0);
    let b1: Vec<_> = patches(&s1, &f1).iter().map(|p| binary_descriptor(p, &pairs)).collect();
    let b2: Vec<_> = patches(&s2, &f2).iter().map(|p| binary_descriptor(p, &pairs)).collect();
    let mut correct = 0;
    let (mut to_partner, mut to_others, mut others) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let nearest = (0..n).min_by_key(|&j| hamming(&b1[i], &b2[j])).unwrap();
        correct += (nearest == i) as usize;
        to_partner += hamming(&b1[i], &b2[i]) as f32 / n as f32;
        for j in (0..n).filter(|&j| j != i) {
            to_others += hamming(&b1[i], &b2[j]) as f32;
            others += 1.0;
        }
    }
    println!("  {:24} binary (256 bits)  {correct:3} ({:3.0} %)", "with θ", pct(correct));
    println!("binary: mean Hamming distance {to_partner:.1} to the partner, {:.1} to the other points", to_others / others);

    // Pictures: the two views with the frames of a few points, and the patches of one point.
    let (f1, f2) = frames(true, 0.0);
    let mut v1 = gray_to_rgb(&img);
    let mut v2 = gray_to_rgb(&img2.map(|v| v / GAIN));
    for k in (0..n).step_by((n / 8).max(1)) {
        draw_frame(&mut v1, f1[k]);
        draw_frame(&mut v2, f2[k]);
    }
    save_rgb(&v1, format!("{dir}/view1.png"));
    save_rgb(&v2, format!("{dir}/view2.png"));
    let k = n / 2;
    let (_, without2) = frames(false, 0.0);
    let show = [sample_patch(&img, f1[k].0, f1[k].1, SIGMA, 0.0, 16), sample_patch(&img2, f2[k].0, f2[k].1, ZOOM * SIGMA, ANGLE, 16).map(|v| v / GAIN), sample_patch(&img2, without2[k].0, without2[k].1, ZOOM * SIGMA, 0.0, 16).map(|v| v / GAIN)];
    let sheet = Image::from_fn(3 * 17 * 8, 16 * 8, |x, y| {
        let (p, xx) = (x / (17 * 8), (x % (17 * 8)) / 8);
        if xx == 16 { 1.0 } else { show[p][(xx, y / 8)] }
    });
    save_gray(&sheet, format!("{dir}/patches.png"));
    println!("wrote the pictures to {dir}/ (patches.png: point {k} in view 1, in view 2 with θ, in view 2 without θ)");
}

/// The frame as a square of side 12σ turned by θ, with a line from the centre along the frame's x axis.
fn draw_frame(img: &mut Image<[f32; 3]>, (x, y, sigma, theta): Frame) {
    let (s, c) = theta.to_radians().sin_cos();
    let h = 6.0 * sigma;
    let to_screen = |u: f32, v: f32| (x + u * c + v * s, y - u * s + v * c);
    let mut put = |(px, py): (f32, f32)| {
        if img.contains(px.round() as isize, py.round() as isize) {
            img[(px.round() as usize, py.round() as usize)] = [1.0, 0.85, 0.0];
        }
    };
    for i in 0..=48 {
        let t = -h + 2.0 * h * i as f32 / 48.0;
        for (u, v) in [(t, -h), (t, h), (-h, t), (h, t)] {
            put(to_screen(u, v));
        }
        if t >= 0.0 {
            put(to_screen(t, 0.0));
        }
    }
}
