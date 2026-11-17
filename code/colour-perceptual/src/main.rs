// Segments the red apple of a synthetic scene by an RGB threshold, by a hue wedge and by
// k-means in RGB and in Lab, and compares each with the true apple (Dice). Pictures in
// out/colour-perceptual/.
//   cargo run -p colour-perceptual                      (your code)
//   cargo run -p colour-perceptual --features solution  (the model solution)

use colour_perceptual::*;
use imgcourse::*;

fn encode(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.0031308 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

/// A synthetic scene, `size × size`, **encoded** sRGB with a little noise: a beige table, a red
/// apple, a yellow fruit, a green leaf, a blue cup and a grey card; a shadow covers the lower
/// right half of the scene, including half of the apple. Also returns the true apple mask.
fn scene(size: usize, noise: f32, seed: u64) -> (Rgb, Image<bool>) {
    let s = size as f32;
    let mut rng = Rng::new(seed);
    let mut apple = Image::new(size, size, false);
    let img = Image::from_fn(size, size, |x, y| {
        let (u, v) = (x as f32 / s, y as f32 / s);
        let disc = |cx: f32, cy: f32, r: f32| ((u - cx).powi(2) + (v - cy).powi(2)).sqrt() < r;
        let colour = if disc(0.3, 0.3, 0.16) {
            apple[(x, y)] = true;
            [0.50, 0.035, 0.03]
        } else if disc(0.72, 0.28, 0.12) {
            [0.62, 0.42, 0.03]
        } else if ((u - 0.28) / 0.18).powi(2) + ((v - 0.75) / 0.08).powi(2) < 1.0 {
            [0.06, 0.25, 0.04]
        } else if (0.58..0.85).contains(&u) && (0.55..0.88).contains(&v) {
            [0.03, 0.07, 0.40]
        } else if (0.05..0.20).contains(&u) && (0.04..0.12).contains(&v) {
            [0.18, 0.18, 0.18]
        } else {
            [0.22, 0.17, 0.12]
        };
        // the shadow: full light above the diagonal u + v = 0.55, a third of it below 0.65
        let light = 1.0 - 0.67 * ((u + v - 0.55) / 0.1).clamp(0.0, 1.0);
        colour.map(|c| encode(c * light) + noise * rng.normal())
    });
    (img, apple)
}

fn dice(a: &Image<bool>, b: &Image<bool>) -> f32 {
    let (mut both, mut na, mut nb) = (0, 0, 0);
    for (x, y) in a.coords() {
        na += a[(x, y)] as usize;
        nb += b[(x, y)] as usize;
        both += (a[(x, y)] && b[(x, y)]) as usize;
    }
    2.0 * both as f32 / (na + nb) as f32
}

fn show(mask: &Image<bool>, img: &Rgb) -> Rgb {
    mask.zip_map(img, |m, c| if m { c } else { c.map(|v| 0.15 * v) })
}

fn main() {
    let dir = "out/colour-perceptual";
    let (img, apple) = scene(256, 0.01, 1);
    save_rgb(&img, format!("{dir}/scene.png"));
    let n_apple = apple.pixels().iter().filter(|&&a| a).count();
    println!("true apple: {n_apple} pixels");

    // An RGB threshold: R′ ≥ 120, G′ ≤ 90, B′ ≤ 90 (codes).
    let rgb_mask = img.map(|c| c[0] >= 120.0 / 255.0 && c[1] <= 90.0 / 255.0 && c[2] <= 90.0 / 255.0);
    save_rgb(&show(&rgb_mask, &img), format!("{dir}/rgb-threshold.png"));
    println!("R threshold:          Dice {:.3}", dice(&rgb_mask, &apple));

    // A hue wedge 340°–20° with saturation ≥ 0.4 and value ≥ 0.1.
    let hue = hue_mask(&img, 340.0, 20.0, 0.4, 0.1);
    save_rgb(&show(&hue, &img), format!("{dir}/hue-wedge.png"));
    println!("hue wedge:            Dice {:.3}", dice(&hue, &apple));

    // The same wedge without floors, on a noisier image: the greys and darks speckle.
    let (noisy, _) = scene(256, 0.03, 2);
    let no_floor = hue_mask(&noisy, 340.0, 20.0, 0.0, 0.0);
    save_rgb(&show(&no_floor, &noisy), format!("{dir}/hue-no-floor.png"));
    let wrong = no_floor.zip_map(&apple, |m, a| m && !a).pixels().iter().filter(|&&w| w).count();
    println!("hue, no floors, noise 0.03: Dice {:.3}, {wrong} pixels outside the apple", dice(&no_floor, &apple));

    // k-means with 4 clusters, in RGB and in Lab, from the same four starting pixels.
    let starts = [(10, 10), (80, 80), (200, 200), (180, 60)];
    let rgb_points: Vec<[f32; 3]> = img.pixels().to_vec();
    let lab_points: Vec<[f32; 3]> = rgb_points.iter().map(|&c| srgb_to_lab(c)).collect();
    for (name, points) in [("rgb", &rgb_points), ("lab", &lab_points)] {
        let init: Vec<[f32; 3]> = starts.iter().map(|&(x, y)| points[y * 256 + x]).collect();
        let (_, labels) = kmeans(points, &init, 20);
        // paint every pixel in its cluster's mean colour (in RGB), and find the best cluster
        let mut mean = [[0.0f32; 3]; 4];
        let mut count = [0.0f32; 4];
        for (i, &l) in labels.iter().enumerate() {
            for d in 0..3 {
                mean[l][d] += rgb_points[i][d];
            }
            count[l] += 1.0;
        }
        let painted = Image::from_fn(256, 256, |x, y| {
            let l = labels[y * 256 + x];
            mean[l].map(|v| v / count[l])
        });
        save_rgb(&painted, format!("{dir}/kmeans-{name}.png"));
        let best = (0..4)
            .map(|k| dice(&Image::from_fn(256, 256, |x, y| labels[y * 256 + x] == k), &apple))
            .fold(0.0f32, f32::max);
        println!("k-means (4) in {name}:   best cluster Dice {best:.3}, cluster sizes {count:?}");
    }

    // Two pairs of colours 40 codes apart (lecture slide "ΔE").
    for (a, b) in [([255.0, 255.0, 0.0], [255.0, 255.0, 40.0]), ([0.0, 80.0, 0.0], [0.0, 120.0, 0.0])] {
        let (la, lb) = (srgb_to_lab(a.map(|v: f32| v / 255.0)), srgb_to_lab(b.map(|v: f32| v / 255.0)));
        println!("{a:?} vs {b:?}: ΔE {:.1}", delta_e(la, lb));
    }
    println!("wrote the pictures to {dir}/");
}
