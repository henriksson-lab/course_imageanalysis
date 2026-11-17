// Builds a shape model of 40 synthetic leaves with your Procrustes, GPA and PCA; pictures in
// out/shape-models/.
//   cargo run -p snakes -- shape-models                      (your code)
//   cargo run -p snakes --features solution -- shape-models  (the model solution)

use imgcourse::*;
use crate::shape_models::*;

const HALF: usize = 12; // landmarks per side; 2 × 12 = 24 in all

/// A leaf from three hidden parameters: `width`, `bend` (of the midrib) and `tip` (where the
/// widest point lies). Landmark 0 is the base, landmark 12 the tip, up one side and back the
/// other: the landmarks correspond by construction.
fn leaf(width: f64, bend: f64, tip: f64) -> Vec<Point> {
    let mut p = Vec::new();
    for side in [1.0, -1.0] {
        for i in 0..HALF {
            let s = if side > 0.0 { i as f64 / HALF as f64 } else { 1.0 - i as f64 / HALF as f64 };
            let half_width = width * s.powf(tip) * (1.0 - s) * 2.5;
            let x = -1.0 + 2.0 * s;
            let y = bend * (s * s - s) * 2.0 + side * half_width;
            p.push([x, y]);
        }
    }
    p
}

/// 40 leaves in random positions, sizes and angles, with a little landmark noise.
fn training_set(rng: &mut Rng) -> Vec<Vec<Point>> {
    (0..40)
        .map(|_| {
            let shape = leaf(0.35 + 0.08 * rng.normal() as f64, 0.15 * rng.normal() as f64, 0.8 + 0.3 * rng.normal() as f64);
            let (scale, angle) = (25.0 + 10.0 * rng.uniform() as f64, std::f64::consts::TAU * rng.uniform() as f64);
            let t = [100.0 + 30.0 * rng.normal() as f64, 100.0 + 30.0 * rng.normal() as f64];
            let pose = Similarity { a: scale * angle.cos(), b: scale * angle.sin(), t };
            apply(pose, &shape).iter().map(|q| [q[0] + 0.4 * rng.normal() as f64, q[1] + 0.4 * rng.normal() as f64]).collect()
        })
        .collect()
}

fn draw(img: &mut Image<[f32; 3]>, p: &[Point], colour: [f32; 3]) {
    for i in 0..p.len() {
        let (a, b) = (p[i], p[(i + 1) % p.len()]);
        let steps = 2 * ((b[0] - a[0]).abs().max((b[1] - a[1]).abs()).ceil() as usize).max(1);
        for s in 0..=steps {
            let t = s as f64 / steps as f64;
            let (x, y) = ((a[0] + t * (b[0] - a[0])).round() as isize, (a[1] + t * (b[1] - a[1])).round() as isize);
            if img.contains(x, y) {
                img[(x as usize, y as usize)] = colour;
            }
        }
    }
}

/// A shape of size 1 (centred) drawn in a 200 × 200 panel.
fn to_panel(p: &[Point]) -> Vec<Point> {
    p.iter().map(|q| [100.0 + 300.0 * q[0], 100.0 + 300.0 * q[1]]).collect()
}

/// The percentage of the variance in each of the first `k` modes.
fn shares(values: &[f64], k: usize) -> Vec<f64> {
    let total: f64 = values.iter().map(|v| v.max(0.0)).sum();
    values[..k].iter().map(|v| 100.0 * v / total).collect()
}

pub fn main() {
    let dir = "out/shape-models";
    let (grey, black, blue, red) = ([0.7; 3], [0.0; 3], [0.1, 0.3, 0.9], [0.9, 0.1, 0.1]);
    let mut rng = Rng::new(7);
    let shapes = training_set(&mut rng);

    // Panel 1: the raw shapes. Panel 2: aligned, with the mean.
    let mut raw = Image::new(200, 200, [1.0f32; 3]);
    for s in &shapes {
        draw(&mut raw, s, grey);
    }
    let (mean, aligned) = gpa(&shapes, 3);
    let spread: f64 = aligned.iter().map(|s| sum_sq(s, &mean)).sum();
    println!("after GPA: sum of squared distances to the mean {spread:.4}");
    let mut al = Image::new(200, 200, [1.0f32; 3]);
    for s in &aligned {
        draw(&mut al, &to_panel(s), grey);
    }
    draw(&mut al, &to_panel(&mean), black);

    // The point distribution model, and the first three modes at ±3 standard deviations.
    let vectors: Vec<Vec<f64>> = aligned.iter().map(|s| flatten(s)).collect();
    let (mean_v, values, modes) = shape_model(&vectors);
    let sh = shares(&values, 6);
    println!("variance per mode (%): {}", sh.iter().map(|v| format!("{v:.1}")).collect::<Vec<_>>().join(", "));
    let mut panels = vec![raw, al];
    for j in 0..3 {
        let mut pic = Image::new(200, 200, [1.0f32; 3]);
        draw(&mut pic, &to_panel(&unflatten(&mean_v)), grey);
        for (sign, colour) in [(-3.0, blue), (3.0, red)] {
            let mut b = vec![0.0; j + 1];
            b[j] = sign * values[j].sqrt();
            draw(&mut pic, &to_panel(&unflatten(&generate(&mean_v, &modes, &b))), colour);
        }
        panels.push(pic);
    }
    let sheet = Image::from_fn(panels.len() * 204 - 4, 200, |x, y| if x % 204 < 200 { panels[x / 204][(x % 204, y)] } else { [0.5; 3] });
    save_rgb(&sheet, format!("{dir}/leaves.png"));

    // Fit a new leaf with the model: how close can 3 modes get, and what does clamping do?
    let new_leaf = normalise(&leaf(0.35, 0.6, 0.8)); // wider and more bent than most
    let target = apply(procrustes(&new_leaf, &mean, true), &new_leaf);
    let x = flatten(&target);
    for clamp in [false, true] {
        let b = project(&x, &mean_v, &modes, &values, 3, clamp);
        let fit = generate(&mean_v, &modes, &b);
        let sd: Vec<String> = b.iter().zip(&values).map(|(bj, l)| format!("{:.2}", bj / l.sqrt())).collect();
        println!("new leaf, clamp {clamp}: b / sqrt(lambda) = [{}], distance to the leaf {:.4}", sd.join(", "), sum_sq(&unflatten(&fit), &target).sqrt());
    }

    // Bad correspondence: 10 of the 40 leaves numbered one landmark off.
    let shifted: Vec<Vec<Point>> = shapes
        .iter()
        .enumerate()
        .map(|(k, s)| if k % 4 == 0 { (0..s.len()).map(|i| s[(i + 1) % s.len()]).collect() } else { s.clone() })
        .collect();
    let (_, aligned2) = gpa(&shifted, 3);
    let (_, values2, _) = shape_model(&aligned2.iter().map(|s| flatten(s)).collect::<Vec<_>>());
    let sh2 = shares(&values2, 3);
    let (t1, t2): (f64, f64) = (values.iter().sum(), values2.iter().sum());
    println!("total variance: {t1:.5} correct, {t2:.5} with the misnumbered leaves");
    println!("10 leaves misnumbered, variance per mode (%): {}", sh2.iter().map(|v| format!("{v:.1}")).collect::<Vec<_>>().join(", "));
    println!("wrote {dir}/leaves.png (raw, aligned with mean, modes 1 to 3 at -3 (blue) and +3 (red) SD)");
}
