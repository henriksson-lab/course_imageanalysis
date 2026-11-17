// Trains your kernel SVM on a ring around a blob, on XOR and on two noisy half-moons, with
// different kernel functions and settings; pictures in out/kernel-trick/.
//   cargo run -p kernel-trick                      (your code)
//   cargo run -p kernel-trick --features solution  (the model solution)

use imgcourse::*;
use kernel_trick::*;

const BLUE: [f32; 3] = [0.25, 0.45, 0.85];
const RED: [f32; 3] = [0.85, 0.3, 0.25];

/// A picture of the feature space `lo..hi` (x1 to the right, x2 up) for a trained SVM: the two
/// classes' regions tinted, the street `−1 ≤ f ≤ 1` in grey, the boundary `f = 0` in black,
/// the points as dots and the support vectors (`α > 0`) with a black ring.
fn plot(points: &[Vec<f32>], labels: &[f32], alpha: &[f32], f: impl Fn(&[f32]) -> f32, lo: [f32; 2], hi: [f32; 2], size: usize) -> Image<[f32; 3]> {
    let scale = [(hi[0] - lo[0]) / size as f32, (hi[1] - lo[1]) / size as f32];
    let values = Image::from_fn(size + 1, size + 1, |px, py| f(&[lo[0] + (px as f32 + 0.5) * scale[0], hi[1] - (py as f32 + 0.5) * scale[1]]));
    let mut img = Image::from_fn(size, size, |px, py| {
        let v = values[(px, py)];
        if v * values[(px + 1, py)] <= 0.0 || v * values[(px, py + 1)] <= 0.0 {
            [0.0; 3]
        } else if v.abs() < 1.0 {
            [0.82; 3]
        } else {
            (if v > 0.0 { RED } else { BLUE }).map(|c| 0.8 + 0.2 * c)
        }
    });
    for i in 0..points.len() {
        let (cx, cy) = ((points[i][0] - lo[0]) / scale[0], (hi[1] - points[i][1]) / scale[1]);
        for dy in -6..=6isize {
            for dx in -6..=6isize {
                let (x, y) = (cx as isize + dx, cy as isize + dy);
                let r2 = dx * dx + dy * dy;
                if !img.contains(x, y) {
                    continue;
                }
                if r2 <= 9 {
                    img[(x as usize, y as usize)] = if r2 >= 7 { [0.0; 3] } else if labels[i] > 0.0 { RED } else { BLUE };
                } else if alpha[i] > 0.0 && (25..=36).contains(&r2) {
                    img[(x as usize, y as usize)] = [0.0; 3];
                }
            }
        }
    }
    img
}

/// Train, print the number of support vectors and errors (on the training points, and on
/// `test` points if any), and save a picture.
#[allow(clippy::too_many_arguments)]
fn run(name: &str, points: &[Vec<f32>], labels: &[f32], test: &[(Vec<f32>, f32)], k: Kernel, c: f32, lo: [f32; 2], hi: [f32; 2]) {
    let (alpha, b) = train_smo(points, labels, k, c, 1e-3, 200_000);
    let f = |x: &[f32]| decision(points, labels, &alpha, b, k, x);
    let support = alpha.iter().filter(|&&a| a > 0.0).count();
    let errors = (0..points.len()).filter(|&i| labels[i] * f(&points[i]) <= 0.0).count();
    let mut line = format!("  {name:<28} {support:>3} support vectors, {errors:>2} of {} training points wrong", points.len());
    if !test.is_empty() {
        let wrong = test.iter().filter(|(x, y)| y * f(x) <= 0.0).count();
        line += &format!(", {:.1} % of new points wrong", 100.0 * wrong as f32 / test.len() as f32);
    }
    println!("{line}");
    save_rgb(&plot(points, labels, &alpha, f, lo, hi, 300), format!("out/kernel-trick/{name}.png"));
}

fn main() {
    let mut rng = Rng::new(2);

    // 1. A ring around a blob (red inside, blue around).
    let (mut points, mut labels) = (Vec::new(), Vec::new());
    for i in 0..60 {
        let (r, y) = if i < 20 { (1.2 * rng.uniform().sqrt(), 1.0) } else { (2.2 + 0.8 * rng.uniform(), -1.0) };
        let t = 2.0 * std::f32::consts::PI * rng.uniform();
        points.push(vec![r * t.cos(), r * t.sin()]);
        labels.push(y);
    }
    println!("ring around a blob:");
    let (lo, hi) = ([-3.5, -3.5], [3.5, 3.5]);
    run("ring-linear", &points, &labels, &[], Kernel::Linear, 10.0, lo, hi);
    run("ring-poly2", &points, &labels, &[], Kernel::Polynomial { c: 1.0, degree: 2 }, 10.0, lo, hi);
    run("ring-rbf", &points, &labels, &[], Kernel::Rbf { gamma: 0.5 }, 10.0, lo, hi);

    // 2. XOR: four clouds in the corners.
    let (mut points, mut labels) = (Vec::new(), Vec::new());
    for (cx, cy, y) in [(1.0, 1.0, -1.0), (-1.0, -1.0, -1.0), (1.0, -1.0, 1.0), (-1.0, 1.0, 1.0)] {
        for _ in 0..15 {
            points.push(vec![cx + 0.35 * rng.normal(), cy + 0.35 * rng.normal()]);
            labels.push(y);
        }
    }
    println!("XOR:");
    let (lo, hi) = ([-2.5, -2.5], [2.5, 2.5]);
    run("xor-linear", &points, &labels, &[], Kernel::Linear, 10.0, lo, hi);
    run("xor-poly2", &points, &labels, &[], Kernel::Polynomial { c: 1.0, degree: 2 }, 10.0, lo, hi);

    // 3. Two noisy half-moons: γ and C. 80 training points, and 2000 new points from the same
    //    recipe to measure how well each setting does on data it has not seen.
    let mut moon = |n: usize| -> Vec<(Vec<f32>, f32)> {
        (0..n)
            .map(|i| {
                let t = std::f32::consts::PI * rng.uniform();
                let (x, y, label) = if i % 2 == 0 { (t.cos(), t.sin(), 1.0) } else { (1.0 - t.cos(), 0.5 - t.sin(), -1.0) };
                (vec![x + 0.3 * rng.normal(), y + 0.3 * rng.normal()], label)
            })
            .collect()
    };
    let train = moon(80);
    let test = moon(2000);
    let points: Vec<Vec<f32>> = train.iter().map(|(x, _)| x.clone()).collect();
    let labels: Vec<f32> = train.iter().map(|(_, y)| *y).collect();
    println!("two half-moons (80 training points, 2000 new points):");
    let (lo, hi) = ([-1.8, -1.5], [2.8, 2.0]);
    for gamma in [0.1, 2.0, 100.0] {
        for c in [0.1, 1000.0] {
            run(&format!("moons-gamma{gamma}-c{c}"), &points, &labels, &test, Kernel::Rbf { gamma }, c, lo, hi);
        }
    }
    println!("wrote the pictures to out/kernel-trick/");
}
