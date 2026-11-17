// Segments a noisy test image with your Chan–Vese and compares it with thresholding;
// pictures in out/chanvese/.
//   cargo run -p chanvese                      (your code)
//   cargo run -p chanvese --features solution  (the model solution)

use chanvese::*;
use imgcourse::*;

const SIZE: usize = 128;

/// The test image without noise (objects 0.7 on 0.3: a ring, a square, a disk with a blurred
/// edge, a small disk and a bar 3 pixels wide) and the true object mask.
fn test_image() -> (Image<f32>, Image<bool>) {
    let mut img = Image::new(SIZE, SIZE, 0.3f32);
    let mut truth = Image::new(SIZE, SIZE, false);
    for (x, y) in img.coords() {
        let (fx, fy) = (x as f32, y as f32);
        let r = |cx: f32, cy: f32| ((fx - cx).powi(2) + (fy - cy).powi(2)).sqrt();
        let ring = (10.0..22.0).contains(&r(34.0, 34.0));
        let square = (76.0..108.0).contains(&fx) && (18.0..50.0).contains(&fy);
        let small = r(100.0, 100.0) < 6.0;
        let bar = (60.0..63.0).contains(&fx) && (70.0..120.0).contains(&fy);
        if ring || square || small || bar {
            img[(x, y)] = 0.7;
            truth[(x, y)] = true;
        }
        // the blurred disk: a soft edge about 10 pixels wide
        let soft = 1.0 / (1.0 + ((r(30.0, 94.0) - 16.0) / 2.5).exp());
        img[(x, y)] += 0.4 * soft;
        if r(30.0, 94.0) < 16.0 {
            truth[(x, y)] = true;
        }
    }
    (img.map(|v| v.min(0.7)), truth)
}

/// The classic start, a grid of small blobs: `φ = sin(πx/8) sin(πy/8)`, negative and positive
/// squares of 8 × 8 pixels. All values lie in [−1, 1], so every pixel can change from the start.
fn blobs() -> Image<f32> {
    let p = std::f32::consts::PI / 8.0;
    Image::from_fn(SIZE, SIZE, |x, y| (p * x as f32).sin() * (p * y as f32).sin())
}

/// Inside and outside play the same role in the energy: call the brighter region the object
/// (flip φ if the inside came out darker).
fn brighter_inside((phi, c1, c2): (Image<f32>, f32, f32)) -> (Image<f32>, f32, f32) {
    if c1 >= c2 { (phi, c1, c2) } else { (phi.map(|v| -v), c2, c1) }
}

fn wrong(inside: impl Fn(usize, usize) -> bool, truth: &Image<bool>) -> usize {
    truth.coords().filter(|&(x, y)| inside(x, y) != truth[(x, y)]).count()
}

/// The noisy image with the inside in blue and the zero level in yellow.
fn picture(img: &Image<f32>, phi: &Image<f32>) -> Image<[f32; 3]> {
    let mut pic = img.map(|v| [v.clamp(0.0, 1.0); 3]);
    for (x, y) in img.coords() {
        let p = phi[(x, y)];
        let edge = p < 0.0
            && [(1, 0), (-1, 0), (0, 1), (0, -1)].iter().any(|&(dx, dy)| {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                phi.contains(nx, ny) && phi[(nx as usize, ny as usize)] >= 0.0
            });
        if edge {
            pic[(x, y)] = [1.0, 0.9, 0.0];
        } else if p < 0.0 {
            pic[(x, y)][2] = (pic[(x, y)][2] + 0.4).min(1.0);
        }
    }
    pic
}

fn sheet(pics: &[Image<[f32; 3]>]) -> Image<[f32; 3]> {
    let w = SIZE + 4;
    Image::from_fn(2 * (pics.len() * w - 4), 2 * SIZE, |x, y| {
        let (x, y) = (x / 2, y / 2);
        if x % w < SIZE { pics[x / w][(x % w, y)] } else { [1.0; 3] }
    })
}

fn main() {
    let dir = "out/chanvese";
    let (clean, truth) = test_image();
    let base = Params { lambda: 0.2, nu: 0.0, dt: 1.0, eps: 1.0 };

    // Noise 0.3: Chan–Vese against thresholding at (c1 + c2)/2.
    let img = samples::add_noise(&clean, 0.3, 1);
    let (phi, c1, c2) = brighter_inside(chan_vese(&img, &blobs(), base, 300));
    let t = (c1 + c2) / 2.0;
    let threshold = img.map(|v| if v > t { -1.0 } else { 1.0 });
    println!("noise 0.3, lambda = 0.2, 300 iterations: c1 = {c1:.2}, c2 = {c2:.2}, {} wrong pixels", wrong(|x, y| phi[(x, y)] < 0.0, &truth));
    println!("noise 0.3, threshold at (c1 + c2)/2 = {t:.2}: {} wrong pixels", wrong(|x, y| img[(x, y)] > t, &truth));
    let inside = phi.map(|v| v < 0.0);
    println!("energy (lambda = 0.2): Chan-Vese {:.0}, threshold {:.0}", energy(&img, &inside, 0.2), energy(&img, &img.map(|v| v > t), 0.2));
    save_rgb(&sheet(&[picture(&img, &threshold), picture(&img, &phi)]), format!("{dir}/noise.png"));

    // Noise 0.15: λ = 0, 0.1 and 1.
    let img = samples::add_noise(&clean, 0.15, 2);
    let mut pics = Vec::new();
    for lambda in [0.0, 0.1, 1.0] {
        let (phi, c1, c2) = brighter_inside(chan_vese(&img, &blobs(), Params { lambda, ..base }, 300));
        let t = (c1 + c2) / 2.0;
        let differ = img.coords().filter(|&(x, y)| (phi[(x, y)] < 0.0) != (img[(x, y)] > t)).count();
        println!(
            "noise 0.15, lambda = {lambda}: {} wrong pixels, {} pixels differ from the threshold at (c1 + c2)/2",
            wrong(|x, y| phi[(x, y)] < 0.0, &truth),
            differ
        );
        pics.push(picture(&img, &phi));
    }
    save_rgb(&sheet(&pics), format!("{dir}/lambda.png"));
    println!("wrote the pictures to {dir}/");
}
