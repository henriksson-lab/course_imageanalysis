// Runs your linear classifiers on small 2D examples and on the pixels of a synthetic
// fluorescence image; pictures in out/classify/.
//   cargo run -p day34 -- classify                      (your code)
//   cargo run -p day34 --features solution -- classify  (the model solution)

use crate::classify::*;
use imgcourse::*;

/// Colours for class −1 (blue) and class +1 (red).
const BLUE: [f32; 3] = [0.25, 0.45, 0.85];
const RED: [f32; 3] = [0.85, 0.3, 0.25];

/// A picture of the 2D feature space `lo..hi` (x1 to the right, x2 up), `size × size` pixels:
/// the background tinted by the sign of `f`, the line `f = 0` in black, and the points as
/// dots in the colour of their label.
fn plot(points: &[Point], labels: &[f32], lo: Point, hi: Point, size: usize, f: impl Fn(Point) -> f32) -> Image<[f32; 3]> {
    let scale = [(hi[0] - lo[0]) / size as f32, (hi[1] - lo[1]) / size as f32];
    let feature = |px: f32, py: f32| [lo[0] + px * scale[0], hi[1] - py * scale[1]];
    let mut img = Image::from_fn(size, size, |px, py| {
        let v = f(feature(px as f32 + 0.5, py as f32 + 0.5));
        // the line: where f changes sign within one pixel
        let right = f(feature(px as f32 + 1.5, py as f32 + 0.5));
        let below = f(feature(px as f32 + 0.5, py as f32 + 1.5));
        if v * right <= 0.0 || v * below <= 0.0 {
            [0.0; 3]
        } else {
            let c = if v > 0.0 { RED } else { BLUE };
            c.map(|c| 0.8 + 0.2 * c)
        }
    });
    for (p, &y) in points.iter().zip(labels) {
        let (cx, cy) = ((p[0] - lo[0]) / scale[0], (hi[1] - p[1]) / scale[1]);
        for dy in -3..=3isize {
            for dx in -3..=3isize {
                let (x, yy) = (cx as isize + dx, cy as isize + dy);
                let r2 = dx * dx + dy * dy;
                if r2 <= 9 && img.contains(x, yy) {
                    img[(x as usize, yy as usize)] = if r2 >= 7 { [0.0; 3] } else if y > 0.0 { RED } else { BLUE };
                }
            }
        }
    }
    img
}

/// Two clouds of `n` points each around the centres `c_plus` and `c_minus`, with standard
/// deviations `sd` along x1 and x2, turned by `angle` (radians).
fn two_clouds(n: usize, c_plus: Point, c_minus: Point, sd: Point, angle: f32, seed: u64) -> (Vec<Point>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let (c, s) = (angle.cos(), angle.sin());
    let (mut points, mut labels) = (Vec::new(), Vec::new());
    for (centre, y) in [(c_plus, 1.0), (c_minus, -1.0)] {
        for _ in 0..n {
            let (u, v) = (sd[0] * rng.normal(), sd[1] * rng.normal());
            points.push([centre[0] + c * u - s * v, centre[1] + s * u + c * v]);
            labels.push(y);
        }
    }
    (points, labels)
}

fn report(name: &str, points: &[Point], labels: &[f32], w: Point, b: f32) {
    println!(
        "  {name}: w = ({:.2}, {:.2}), b = {:.2}, {} of {} wrong, smallest y·f/‖w‖ = {:.3}",
        w[0],
        w[1],
        b,
        count_errors(points, labels, w, b),
        points.len(),
        smallest_margin(points, labels, w, b)
    );
}

pub fn main() {
    let dir = "out/classify";

    // 1. The six points of B3: nearest class mean and the perceptron.
    let b3 = [[3.0, 3.0], [4.0, 2.0], [5.0, 4.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0]];
    let b3_labels = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0];
    println!("B3 (six points):");
    let (w, b) = nearest_mean(&b3, &b3_labels);
    report("nearest class mean", &b3, &b3_labels, w, b);
    save_rgb(&plot(&b3, &b3_labels, [-1.0, -1.0], [6.0, 6.0], 300, |x| decision(w, b, x)), format!("{dir}/b3-nearest-mean.png"));
    let (w, b, updates) = perceptron(&b3, &b3_labels, 1000);
    report(&format!("perceptron ({updates} updates)"), &b3, &b3_labels, w, b);
    save_rgb(&plot(&b3, &b3_labels, [-1.0, -1.0], [6.0, 6.0], 300, |x| decision(w, b, x)), format!("{dir}/b3-perceptron.png"));

    // 2. Two long parallel clouds, shifted along their length: nearest mean is not the best line.
    let (points, labels) = two_clouds(30, [5.8, 6.8], [4.5, 4.0], [2.0, 0.3], 0.5, 3);
    println!("Two long clouds (60 points):");
    let (w, b) = nearest_mean(&points, &labels);
    report("nearest class mean", &points, &labels, w, b);
    save_rgb(&plot(&points, &labels, [0.0, 0.0], [10.0, 10.0], 300, |x| decision(w, b, x)), format!("{dir}/long-nearest-mean.png"));
    let (w, b, updates) = perceptron(&points, &labels, 10000);
    report(&format!("perceptron ({updates} updates)"), &points, &labels, w, b);
    save_rgb(&plot(&points, &labels, [0.0, 0.0], [10.0, 10.0], 300, |x| decision(w, b, x)), format!("{dir}/long-perceptron.png"));

    // 3. XOR: four clouds in the corners; no line works.
    let mut rng = Rng::new(5);
    let (mut xor, mut xor_labels) = (Vec::new(), Vec::new());
    for (cx, cy, y) in [(2.5, 2.5, -1.0), (7.5, 7.5, -1.0), (7.5, 2.5, 1.0), (2.5, 7.5, 1.0)] {
        for _ in 0..15 {
            xor.push([cx + 0.8 * rng.normal(), cy + 0.8 * rng.normal()]);
            xor_labels.push(y);
        }
    }
    println!("XOR (60 points):");
    let (w, b) = nearest_mean(&xor, &xor_labels);
    report("nearest class mean", &xor, &xor_labels, w, b);
    let (w, b, updates) = perceptron(&xor, &xor_labels, 1000);
    report(&format!("perceptron (gave up after {updates} updates)"), &xor, &xor_labels, w, b);
    save_rgb(&plot(&xor, &xor_labels, [0.0, 0.0], [10.0, 10.0], 300, |x| decision(w, b, x)), format!("{dir}/xor-perceptron.png"));

    // 4. Pixels are points: (grey value, mean of the 5 × 5 neighbourhood) of a synthetic
    //    fluorescence image; label +1 = nucleus, −1 = background. Train on 200 random pixels.
    let (img, truth) = samples::nuclei(128, 12, 7);
    let local_mean = img.map_xy(|x, y, _| {
        let mut sum = 0.0;
        for dy in -2..=2isize {
            for dx in -2..=2isize {
                sum += img.get_border(x as isize + dx, y as isize + dy, Border::Mirror);
            }
        }
        sum / 25.0
    });
    let feature = |x: usize, y: usize| [img[(x, y)], local_mean[(x, y)]];
    let mut rng = Rng::new(11);
    let (mut train, mut train_labels) = (Vec::new(), Vec::new());
    for _ in 0..200 {
        let (x, y) = (rng.below(128), rng.below(128));
        train.push(feature(x, y));
        train_labels.push(if truth[(x, y)] > 0 { 1.0 } else { -1.0 });
    }
    println!("Pixels of the synthetic nuclei image (200 training pixels):");
    let (w, b) = nearest_mean(&train, &train_labels);
    report("nearest class mean", &train, &train_labels, w, b);
    let result = Image::from_fn(128, 128, |x, y| decision(w, b, feature(x, y)) > 0.0);
    let correct = img.coords().filter(|&(x, y)| result[(x, y)] == (truth[(x, y)] > 0)).count();
    println!("  on all 16384 pixels: {:.1} % correct", 100.0 * correct as f32 / 16384.0);
    save_gray(&img, format!("{dir}/nuclei.png"));
    save_gray(&result.map(|r| if r { 1.0 } else { 0.0 }), format!("{dir}/nuclei-nearest-mean.png"));
    save_rgb(&plot(&train, &train_labels, [0.0, 0.0], [1.2, 1.2], 300, |x| decision(w, b, x)), format!("{dir}/nuclei-feature-space.png"));
    println!("wrote the pictures to {dir}/");
}
