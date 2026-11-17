// Runs your snake on a noisy disk and on the nucleus of the micrograph; pictures in out/snakes/.
//   cargo run -p snakes -- snakes                      (your code)
//   cargo run -p snakes --features solution -- snakes  (the model solution)

use imgcourse::*;
use crate::snakes::*;

/// Mean, smallest and largest distance of the beads from `(cx, cy)`.
fn radii(points: &[Point], cx: f32, cy: f32) -> (f32, f32, f32) {
    let r: Vec<f32> = points.iter().map(|p| ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt()).collect();
    let mean = r.iter().sum::<f32>() / r.len() as f32;
    (mean, r.iter().copied().fold(f32::INFINITY, f32::min), r.iter().copied().fold(0.0, f32::max))
}

/// Draw the closed polygon through `points` (and a dot on every bead) onto `img`.
fn draw(img: &mut Image<[f32; 3]>, points: &[Point], colour: [f32; 3], dashed: bool) {
    let n = points.len();
    for i in 0..n {
        let (a, b) = (points[i], points[(i + 1) % n]);
        let steps = ((b[0] - a[0]).abs().max((b[1] - a[1]).abs()) * 2.0).ceil().max(1.0) as usize;
        for s in 0..=steps {
            if dashed && (s / 4) % 2 == 1 {
                continue;
            }
            let t = s as f32 / steps as f32;
            let (x, y) = ((a[0] + t * (b[0] - a[0])).round() as isize, (a[1] + t * (b[1] - a[1])).round() as isize);
            if img.contains(x, y) {
                img[(x as usize, y as usize)] = colour;
            }
        }
    }
    if !dashed {
        for p in points {
            let (x, y) = (p[0].round() as isize, p[1].round() as isize);
            if img.contains(x, y) {
                img[(x as usize, y as usize)] = [1.0, 0.5, 0.0];
            }
        }
    }
}

/// Twice as large, so that the beads can be seen.
fn enlarge(img: &Image<[f32; 3]>) -> Image<[f32; 3]> {
    Image::from_fn(2 * img.width(), 2 * img.height(), |x, y| img[(x / 2, y / 2)])
}

pub fn main() {
    let dir = "out/snakes";
    let (yellow, cyan, red) = ([1.0, 0.9, 0.0], [0.0, 0.9, 1.0], [1.0, 0.2, 0.2]);

    // A bright disk of radius 30 on a 128 × 128 image, with a little noise.
    let disk = Image::from_fn(128, 128, |x, y| {
        let r = ((x as f32 - 64.0).powi(2) + (y as f32 - 64.0).powi(2)).sqrt();
        if r < 30.0 { 0.8 } else { 0.2 }
    });
    let disk = samples::add_noise(&disk, 0.05, 1);
    let f = edge_map(&disk, 2.0);
    save_gray(&f, format!("{dir}/edge-map.png"));

    // α = 0.1, β = 0.05, w = 2, τ = 1: from a circle of radius 45 onto the edge.
    let start = circle(64.0, 64.0, 45.0, 40);
    let mut pic = gray_to_rgb(&disk);
    draw(&mut pic, &start, yellow, true);
    let mut s = start.clone();
    for step in 1..=300 {
        s = snake_step(&s, &f, 0.1, 0.05, 2.0, 1.0);
        if step % 100 == 0 {
            let (mean, lo, hi) = radii(&s, 64.0, 64.0);
            println!("tau = 1, step {step}: mean radius {mean:.2} (beads between {lo:.2} and {hi:.2})");
            if step == 100 {
                draw(&mut pic, &s, cyan, false);
            }
        }
    }
    draw(&mut pic, &s, yellow, false);
    save_rgb(&enlarge(&pic), format!("{dir}/disk.png"));

    // α ten times larger: the springs win.
    let mut s = start.clone();
    for _ in 0..500 {
        s = snake_step(&s, &f, 1.0, 0.05, 2.0, 0.3);
    }
    println!("alpha = 1 (tau = 0.3), 500 steps: mean radius {:.2}", radii(&s, 64.0, 64.0).0);

    // τ = 2: the explicit step is unstable (τ (4α + 16β) = 2.4 > 2).
    let mut pic = gray_to_rgb(&disk);
    let mut s = start.clone();
    for step in 1..=60 {
        s = snake_step(&s, &f, 0.1, 0.05, 2.0, 2.0);
        let (mean, lo, hi) = radii(&s, 64.0, 64.0);
        if step % 10 == 0 {
            println!("tau = 2, step {step}: mean radius {mean:.2}, beads between {lo:.2} and {hi:.2}");
        }
        if step == 30 {
            draw(&mut pic, &s, cyan, false);
        }
        if step == 40 {
            draw(&mut pic, &s, red, false);
        }
        if hi > 1000.0 {
            println!("tau = 2: the snake has exploded after {step} steps");
            break;
        }
    }
    save_rgb(&enlarge(&pic), format!("{dir}/explode.png"));

    // The nucleus of the micrograph, at half size (128 × 128).
    let cell = samples::cell();
    let small = Image::from_fn(128, 128, |x, y| {
        (cell[(2 * x, 2 * y)] + cell[(2 * x + 1, 2 * y)] + cell[(2 * x, 2 * y + 1)] + cell[(2 * x + 1, 2 * y + 1)]) / 4.0
    });
    let f = edge_map(&small, 3.0);
    let start = circle(74.0, 57.0, 44.0, 60);
    let mut s = start.clone();
    for _ in 0..500 {
        s = snake_step(&s, &f, 0.1, 0.05, 2.0, 1.0);
    }
    let mut pic = gray_to_rgb(&small);
    draw(&mut pic, &start, yellow, true);
    draw(&mut pic, &s, yellow, false);
    save_rgb(&enlarge(&pic), format!("{dir}/nucleus.png"));
    println!("wrote the pictures to {dir}/");
}
