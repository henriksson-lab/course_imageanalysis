// Trains your kernel SVM on a ring around a blob, on XOR and on two noisy half-moons, with
// different kernel functions and settings; pictures in out/kernel-trick/.
//   cargo run -p day34 -- kernel-trick                      (your code)
//   cargo run -p day34 --features solution -- kernel-trick  (the model solution)

use imgcourse::*;
use crate::kernel_trick::*;
use crate::common::plot;

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
    save_rgb(&plot(points, labels, &alpha.iter().map(|&a| a > 0.0).collect::<Vec<_>>(), true, f, lo, hi, 300), format!("out/kernel-trick/{name}.png"));
}

pub fn main() {
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
