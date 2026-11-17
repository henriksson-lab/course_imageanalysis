// Trains a shape model and a profile model on synthetic nuclei, then runs your active shape
// model search on a noisy nucleus, with and without the shape constraint; pictures in out/asm/.
//   cargo run -p day30 -- asm                      (your code)
//   cargo run -p day30 --features solution -- asm  (the model solution)

use crate::asm::*;
use imgcourse::*;

const N: usize = 32; // landmarks
const K: usize = 3; // half-length of the profile model
const M: usize = 6; // search range along the normal
const SMOOTH: f32 = 1.5; // smoothing of the noisy image before the search

/// A nucleus outline in the model frame (size about 1) from three hidden parameters:
/// elongation `e`, banana bend `k` and a three-lobed wobble `w`. Landmark `i` sits at the angle
/// `2πi/N`, so the landmarks correspond by construction.
fn nucleus(e: f32, k: f32, w: f32) -> Vec<Point> {
    (0..N)
        .map(|i| {
            let t = 2.0 * std::f32::consts::PI * i as f32 / N as f32;
            let r = 1.0 + w * (3.0 * t).cos();
            let (x, y) = ((1.0 + e) * r * t.cos(), (1.0 - e) * r * t.sin());
            [x, y + k * (x * x - 0.5)]
        })
        .collect()
}

fn random_nucleus(rng: &mut Rng) -> Vec<Point> {
    nucleus(0.15 + 0.08 * rng.normal(), 0.15 * rng.normal(), 0.04 * rng.normal())
}

/// PCA of the training outlines (all in the model frame already, so no alignment is needed).
fn train_shape_model(shapes: &[Vec<Point>], modes: usize) -> ShapeModel {
    let vectors: Vec<Vec<f64>> = shapes.iter().map(|s| to_vector(s).iter().map(|&v| v as f64).collect()).collect();
    let (s, d) = (vectors.len() as f64, 2 * N);
    let mean: Vec<f64> = (0..d).map(|i| vectors.iter().map(|v| v[i]).sum::<f64>() / s).collect();
    let cov: Vec<Vec<f64>> =
        (0..d).map(|i| (0..d).map(|j| vectors.iter().map(|v| (v[i] - mean[i]) * (v[j] - mean[j])).sum::<f64>() / s).collect()).collect();
    let (values, vecs) = linalg::sym_eigen(&cov);
    let total: f64 = values.iter().sum();
    println!("shape model: variance in modes 1-4: {:.1}, {:.1}, {:.1}, {:.1} %", 100.0 * values[0] / total, 100.0 * values[1] / total, 100.0 * values[2] / total, 100.0 * values[3] / total);
    ShapeModel {
        mean: mean.iter().map(|&v| v as f32).collect(),
        modes: vecs[..modes].iter().map(|v| v.iter().map(|&x| x as f32).collect()).collect(),
        values: values[..modes].iter().map(|&v| v as f32).collect(),
    }
}

/// A grey image of the outline: `inside` within the polygon, `outside` elsewhere, blurred a
/// little (the optics).
fn render(outline: &[Point], inside: f32, outside: f32) -> Image<f32> {
    let sharp = Image::from_fn(128, 128, |x, y| {
        let (px, py) = (x as f32, y as f32);
        let mut crossings = 0;
        for i in 0..outline.len() {
            let (a, b) = (outline[i], outline[(i + 1) % outline.len()]);
            if (a[1] > py) != (b[1] > py) && px < a[0] + (py - a[1]) / (b[1] - a[1]) * (b[0] - a[0]) {
                crossings += 1;
            }
        }
        if crossings % 2 == 1 { inside } else { outside }
    });
    blur(&sharp, 1.2)
}

/// Gaussian blur of width `sigma` (unit "Filtering & convolution").
fn blur(img: &Image<f32>, sigma: f32) -> Image<f32> {
    let r = (3.0 * sigma).ceil() as isize;
    let w: Vec<f32> = (-r..=r).map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let total: f32 = w.iter().sum();
    let pass = |img: &Image<f32>, dx: isize, dy: isize| {
        img.map_xy(|x, y, _| (-r..=r).map(|d| w[(d + r) as usize] / total * img.get_border(x as isize + d * dx, y as isize + d * dy, Border::Clamp)).sum())
    };
    pass(&pass(img, 1, 0), 0, 1)
}

/// The normalised derivative profile of half-length `k` at `p` along `n` (as in `suggest`).
fn profile_at(img: &Image<f32>, p: Point, n: Point, k: isize) -> Vec<f32> {
    let at = |s: f32| bilinear(img, p[0] + s * n[0], p[1] + s * n[1]);
    let g: Vec<f32> = (-k..=k).map(|o| at(o as f32 + 0.5) - at(o as f32 - 0.5)).collect();
    let total: f32 = g.iter().map(|v| v.abs()).sum::<f32>().max(1e-12);
    g.iter().map(|v| v / total).collect()
}

fn rms(a: &[Point], b: &[Point]) -> f32 {
    (a.iter().zip(b).map(|(p, q)| (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sum::<f32>() / a.len() as f32).sqrt()
}

/// The mean distance from the landmarks `a` to the closed polygon `outline` (sliding along
/// the outline does not count).
fn outline_distance(a: &[Point], outline: &[Point]) -> f32 {
    let seg = |p: Point, u: Point, v: Point| {
        let (dx, dy) = (v[0] - u[0], v[1] - u[1]);
        let t = (((p[0] - u[0]) * dx + (p[1] - u[1]) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
        ((p[0] - u[0] - t * dx).powi(2) + (p[1] - u[1] - t * dy).powi(2)).sqrt()
    };
    let n = outline.len();
    a.iter().map(|&p| (0..n).map(|i| seg(p, outline[i], outline[(i + 1) % n])).fold(f32::INFINITY, f32::min)).sum::<f32>() / a.len() as f32
}

fn draw(img: &mut Image<[f32; 3]>, p: &[Point], colour: [f32; 3], dashed: bool) {
    for i in 0..p.len() {
        let (a, b) = (p[i], p[(i + 1) % p.len()]);
        let steps = ((b[0] - a[0]).abs().max((b[1] - a[1]).abs()) * 2.0).ceil().max(1.0) as usize;
        for s in 0..=steps {
            if dashed && (s / 3) % 2 == 1 {
                continue;
            }
            let t = s as f32 / steps as f32;
            let (x, y) = ((a[0] + t * (b[0] - a[0])).round() as isize, (a[1] + t * (b[1] - a[1])).round() as isize);
            if img.contains(x, y) {
                img[(x as usize, y as usize)] = colour;
            }
        }
    }
}

pub fn main() {
    let dir = "out/asm";
    let mut rng = Rng::new(3);

    // Training: 40 outlines for the shape model, 10 clean images for the profile model.
    let shapes: Vec<Vec<Point>> = (0..40).map(|_| random_nucleus(&mut rng)).collect();
    let model = train_shape_model(&shapes, 3);
    let mut profile = vec![0.0; 2 * K + 1];
    for s in &shapes[..10] {
        let outline = apply(Pose { a: 22.0, b: 0.0, t: [64.0, 64.0] }, s);
        let img = render(&outline, 0.75, 0.25);
        for (p, n) in outline.iter().zip(normals(&outline)) {
            for (acc, v) in profile.iter_mut().zip(profile_at(&img, *p, n, K as isize)) {
                *acc += v / (10 * N) as f32;
            }
        }
    }
    println!("trained profile (inside to outside): {}", profile.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>().join(", "));

    // A new nucleus in the image: pose = 22 pixels, turned by 0.3 rad, at the centre.
    let truth_pose = Pose { a: 22.0 * 0.3f32.cos(), b: 22.0 * 0.3f32.sin(), t: [64.0, 64.0] };
    let truth = apply(truth_pose, &nucleus(0.25, 0.2, 0.03));
    // Start: the mean shape 10 pixels right, 4 up, turned 10° more and 5 % too small.
    let turn = 0.3 + 10f32.to_radians();
    let start_pose = Pose { a: 0.95 * 22.0 * turn.cos(), b: 0.95 * 22.0 * turn.sin(), t: [74.0, 60.0] };
    let start = apply(start_pose, &to_points(&model.mean));

    let mut panels = Vec::new();
    for case in ["noise", "occlusion"] {
        let mut clean = render(&truth, 0.75, 0.25);
        if case == "occlusion" {
            for (x, y) in clean.coords().collect::<Vec<_>>() {
                if (40..48).contains(&y) {
                    clean[(x, y)] = 0.1; // a dark bar across the top of the nucleus
                }
            }
        }
        let noisy = samples::add_noise(&clean, 0.25, 11);
        let img = blur(&noisy, SMOOTH); // the search reads a lightly smoothed image
        // The active shape model.
        let (mut pose, mut b) = (start_pose, vec![0.0; 3]);
        for _ in 0..12 {
            (pose, b) = asm_iteration(&img, &model, pose, &b, M, &profile);
        }
        let fit = apply(pose, &to_points(&model_shape(&model, &b)));
        // Suggestions only: every landmark follows its own suggestion.
        let mut free = start.clone();
        for _ in 0..12 {
            free = suggest(&img, &free, M, &profile);
        }
        let limits: Vec<String> = b.iter().zip(&model.values).map(|(bj, l)| format!("{:.2}", bj / l.sqrt())).collect();
        println!(
            "{case}: RMS landmark error at the start {:.2} px, ASM after 12 iterations {:.2} px (b / sqrt(lambda) = [{}]), suggestions only {:.2} px",
            rms(&start, &truth),
            rms(&fit, &truth),
            limits.join(", "),
            rms(&free, &truth)
        );
        println!(
            "    mean distance of the landmarks to the true outline: start {:.2} px, ASM {:.2} px, suggestions only {:.2} px",
            outline_distance(&start, &truth),
            outline_distance(&fit, &truth),
            outline_distance(&free, &truth)
        );
        let mut pic = gray_to_rgb(&noisy.map(|v| v.clamp(0.0, 1.0)));
        draw(&mut pic, &truth, [0.2, 0.9, 0.2], true);
        draw(&mut pic, &free, [1.0, 0.2, 0.2], false);
        draw(&mut pic, &fit, [1.0, 0.9, 0.0], false);
        panels.push(pic);
    }
    let sheet = Image::from_fn(2 * (2 * 128 + 4), 2 * 128, |x, y| {
        let (x, y) = (x / 2, y / 2);
        if x < 128 { panels[0][(x, y)] } else if x < 132 { [1.0; 3] } else { panels[1][(x - 132, y)] }
    });
    save_rgb(&sheet, format!("{dir}/search.png"));
    println!("wrote {dir}/search.png (green dashed: truth, yellow: ASM, red: suggestions only)");
}
