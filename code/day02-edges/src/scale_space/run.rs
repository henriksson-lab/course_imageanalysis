// Runs blob detection over scales on a synthetic image, and builds the pyramids of the
// course micrograph; results in out/scale-space/.
//   cargo run -p day02 -- scale-space                      (your code)
//   cargo run -p day02 --features solution -- scale-space  (the model solution)

use imgcourse::*;
use crate::scale_space::*;

/// A 256 × 256 test image: bright discs of radii 3 to 16, a touching pair of radius 10 and an
/// elongated blob (26 × 5), on a background that brightens to the right, with noise.
/// Returns the image and the true discs (x, y, r).
fn blob_scene() -> (Image<f32>, Vec<(f32, f32, f32)>) {
    let discs = vec![
        (30.0, 30.0, 3.0),
        (80.0, 30.0, 5.0),
        (140.0, 35.0, 8.0),
        (210.0, 45.0, 12.0),
        (60.0, 110.0, 16.0),
        (150.0, 110.0, 6.0),
        (200.0, 130.0, 4.0),
        (50.0, 205.0, 10.0),
        (70.0, 205.0, 10.0), // touches the one before
    ];
    let clean = Image::from_fn(256, 256, |x, y| {
        let (xf, yf) = (x as f32, y as f32);
        let mut v = 0.1 + 0.25 * xf / 256.0;
        if discs.iter().any(|&(cx, cy, r)| (xf - cx).powi(2) + (yf - cy).powi(2) <= r * r) {
            v += 0.5;
        }
        if (150.0..176.0).contains(&xf) && (200.0..205.0).contains(&yf) {
            v += 0.5; // the elongated blob
        }
        v
    });
    (samples::add_noise(&clean, 0.04, 7), discs)
}

/// Draw a circle of radius `r` around (cx, cy) in `colour`.
fn circle(img: &mut Image<[f32; 3]>, cx: f32, cy: f32, r: f32, colour: [f32; 3]) {
    let steps = (8.0 * r).max(24.0) as usize;
    for i in 0..steps {
        let t = 2.0 * std::f32::consts::PI * i as f32 / steps as f32;
        let (x, y) = ((cx + r * t.cos()).round() as isize, (cy + r * t.sin()).round() as isize);
        if img.contains(x, y) {
            img[(x as usize, y as usize)] = colour;
        }
    }
}

/// Pictures side by side, top-aligned, 4 pixels apart, on white.
fn side_by_side(pictures: &[Image<f32>]) -> Image<f32> {
    let w: usize = pictures.iter().map(|p| p.width() + 4).sum();
    let h = pictures.iter().map(|p| p.height()).max().unwrap_or(0);
    let mut out = Image::new(w, h, 1.0f32);
    let mut x0 = 0;
    for p in pictures {
        for (x, y) in p.coords() {
            out[(x0 + x, y)] = p[(x, y)];
        }
        x0 += p.width() + 4;
    }
    out
}

/// How many of the true discs have a detection within max(3, r/2) pixels of their centre.
fn hits(blobs: &[Blob], truth: &[(f32, f32, f32)]) -> usize {
    let near = |&(x, y, r): &(f32, f32, f32)| blobs.iter().any(|b| (b.x as f32 - x).hypot(b.y as f32 - y) <= (r / 2.0).max(3.0));
    truth.iter().filter(|t| near(t)).count()
}

pub fn main() {
    let dir = "out/scale-space";

    // Blob detection over 16 scales, σ = 1 … 16, spaced evenly on a log scale.
    let (img, truth) = blob_scene();
    save_gray(&img, format!("{dir}/blobs-input.png"));
    let sigmas: Vec<f32> = (0..16).map(|k| 16f32.powf(k as f32 / 15.0)).collect();
    let stack: Vec<Image<f32>> = sigmas.iter().map(|&s| normalised_log(&img, s)).collect();
    for k in [3, 8, 13] {
        save_gray_signed(&stack[k], format!("{dir}/log-sigma{:.1}.png", sigmas[k]));
    }
    let blobs = detect_blobs(&stack, &sigmas, 0.22);
    println!("{} detections over σ = 1 … 16 (threshold 0.22):", blobs.len());
    let mut drawn = gray_to_rgb(&img);
    for b in &blobs {
        circle(&mut drawn, b.x as f32, b.y as f32, 2f32.sqrt() * b.sigma, [1.0, 0.3, 0.0]);
        println!("  ({:3}, {:3})  σ = {:5.2}  radius {:5.2}  response {:.2}", b.x, b.y, b.sigma, 2f32.sqrt() * b.sigma, b.response);
    }
    save_rgb(&drawn, format!("{dir}/blobs-detected.png"));
    println!("{} of the {} discs hit", hits(&blobs, &truth), truth.len());
    print!("true discs (x, y, r):");
    for (x, y, r) in &truth {
        print!(" ({x}, {y}, {r})");
    }
    println!();

    // The raw (not normalised) LoG: divide by σ² again, and detect over the same scales.
    let raw: Vec<Image<f32>> = stack.iter().zip(&sigmas).map(|(l, s)| l.map(|v| v / (s * s))).collect();
    let raw_blobs = detect_blobs(&raw, &sigmas, 0.12);
    let at_smallest = raw_blobs.iter().filter(|b| b.sigma == sigmas[0]).count();
    println!("raw LoG (threshold 0.12): {} detections, {} of them at σ = 1, {} of the {} discs hit", raw_blobs.len(), at_smallest, hits(&raw_blobs, &truth), truth.len());

    // The DoG next to the normalised LoG at σ = 4.
    let d = dog(&img, 4.0, 1.2);
    let l = normalised_log(&img, 4.0);
    let (cx, cy) = (140, 35);
    println!("at ({cx}, {cy}), σ = 4, k = 1.2: DoG {:.4}, −(k − 1) × normalised LoG {:.4}", d[(cx, cy)], -0.2 * l[(cx, cy)]);

    // Pyramids of the micrograph.
    let cell = samples::cell();
    let mut gauss = vec![cell.clone()];
    for _ in 0..4 {
        let next = downsample(gauss.last().unwrap());
        gauss.push(next);
    }
    save_gray(&side_by_side(&gauss), format!("{dir}/gaussian-pyramid.png"));
    let lap = laplacian_pyramid(&cell, 5);
    // Show each detail level with 0 as mid-grey, scaled by the same factor.
    let m = lap[..4].iter().flat_map(|l| l.pixels()).fold(0.0f32, |m, v| m.max(v.abs()));
    let mut shown: Vec<Image<f32>> = lap[..4].iter().map(|l| l.map(|v| 0.5 + 0.5 * v / m)).collect();
    shown.push(lap[4].clone());
    save_gray(&side_by_side(&shown), format!("{dir}/laplacian-pyramid.png"));
    let n: usize = gauss.iter().map(|g| g.pixels().len()).sum();
    let back = reconstruct(&lap);
    let err = back.zip_map(&cell, |a, b| (a - b).abs()).max();
    println!("Gaussian pyramid, 5 levels: {n} pixels for {} in the image; Laplacian pyramid rebuilt with largest error {err:.1e}", cell.pixels().len());
    println!("wrote the results to {dir}/");
}
