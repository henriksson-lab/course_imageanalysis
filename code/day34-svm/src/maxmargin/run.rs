// Trains your linear SVM on the lecture's example and on two clouds with an outlier, for
// several values of C; pictures of the street in out/maxmargin/.
//   cargo run -p day34 -- maxmargin                      (your code)
//   cargo run -p day34 --features solution -- maxmargin  (the model solution)

use imgcourse::*;
use crate::maxmargin::*;

const BLUE: [f32; 3] = [0.25, 0.45, 0.85];
const RED: [f32; 3] = [0.85, 0.3, 0.25];

/// A picture of the feature space `lo..hi` (x1 to the right, x2 up): the two sides of the
/// line tinted, the street `−1 ≤ f ≤ 1` in grey, the line `f = 0` in black, the points as dots
/// and the support vectors (`y f ≤ 1`) with a black ring.
fn plot(points: &[Point], labels: &[f32], w: Point, b: f32, lo: Point, hi: Point, size: usize) -> Image<[f32; 3]> {
    let scale = [(hi[0] - lo[0]) / size as f32, (hi[1] - lo[1]) / size as f32];
    let f = |px: f32, py: f32| w[0] * (lo[0] + px * scale[0]) + w[1] * (hi[1] - py * scale[1]) + b;
    let mut img = Image::from_fn(size, size, |px, py| {
        let (px, py) = (px as f32 + 0.5, py as f32 + 0.5);
        let v = f(px, py);
        if v * f(px + 1.0, py) <= 0.0 || v * f(px, py + 1.0) <= 0.0 {
            [0.0; 3]
        } else if v.abs() < 1.0 {
            [0.82; 3]
        } else {
            (if v > 0.0 { RED } else { BLUE }).map(|c| 0.8 + 0.2 * c)
        }
    });
    let m = margins(points, labels, w, b);
    for ((p, &y), &m) in points.iter().zip(labels).zip(&m) {
        let (cx, cy) = ((p[0] - lo[0]) / scale[0], (hi[1] - p[1]) / scale[1]);
        let support = m <= 1.01;
        for dy in -6..=6isize {
            for dx in -6..=6isize {
                let (x, yy) = (cx as isize + dx, cy as isize + dy);
                let r2 = dx * dx + dy * dy;
                if !img.contains(x, yy) {
                    continue;
                }
                if r2 <= 9 {
                    img[(x as usize, yy as usize)] = if r2 >= 7 { [0.0; 3] } else if y > 0.0 { RED } else { BLUE };
                } else if support && (25..=36).contains(&r2) {
                    img[(x as usize, yy as usize)] = [0.0; 3];
                }
            }
        }
    }
    img
}

fn report(points: &[Point], labels: &[f32], c: f32, w: Point, b: f32) {
    let m = margins(points, labels, w, b);
    let support = m.iter().filter(|&&m| m <= 1.01).count();
    let errors = m.iter().filter(|&&m| m <= 0.0).count();
    let width = 2.0 / (w[0] * w[0] + w[1] * w[1]).sqrt();
    println!(
        "  C = {c:>6}: w = ({:.3}, {:.3}), b = {:.3}, width {:.2}, {support} support vectors, {errors} errors, objective {:.3}",
        w[0],
        w[1],
        b,
        width,
        objective(points, labels, w, b, c)
    );
}

pub fn main() {
    let dir = "out/maxmargin";

    // 1. The four losses of B3.
    println!("losses [0–1, hinge, logistic, squared] of B3:");
    for m in [3.0, 1.5, 1.0, 0.2, -0.5, -3.0] {
        let l = losses(m);
        println!("  y f = {m:>4}: {:.2} {:.2} {:.2} {:.2}", l[0], l[1], l[2], l[3]);
    }

    // 2. The lecture's example, for decreasing C.
    let lecture = [[0.0, 0.0], [0.0, 2.0], [2.0, 1.0], [3.0, 0.0]];
    let lecture_labels = [-1.0, -1.0, 1.0, 1.0];
    println!("the lecture's example:");
    for c in [1000.0, 1.0, 0.32, 0.1] {
        let (w, b) = train(&lecture, &lecture_labels, c, 20000);
        report(&lecture, &lecture_labels, c, w, b);
        save_rgb(&plot(&lecture, &lecture_labels, w, b, [-1.5, -1.5], [4.5, 3.5], 300), format!("{dir}/lecture-c{c}.png"));
    }

    // 3. Two clouds and one red outlier close to the blue cloud.
    let mut rng = Rng::new(6);
    let (mut points, mut labels) = (Vec::new(), Vec::new());
    for (cx, cy, y) in [(3.0, 3.0, -1.0), (7.0, 7.0, 1.0)] {
        for _ in 0..25 {
            points.push([cx + 0.7 * rng.normal(), cy + 0.7 * rng.normal()]);
            labels.push(y);
        }
    }
    points.push([4.8, 2.4]);
    labels.push(1.0);
    println!("two clouds and an outlier (51 points):");
    for c in [100.0, 1.0, 0.1, 0.01] {
        let (w, b) = train(&points, &labels, c, 100000);
        report(&points, &labels, c, w, b);
        save_rgb(&plot(&points, &labels, w, b, [0.0, 0.0], [10.0, 10.0], 300), format!("{dir}/outlier-c{c}.png"));
    }
    println!("wrote the pictures to {dir}/");
}
