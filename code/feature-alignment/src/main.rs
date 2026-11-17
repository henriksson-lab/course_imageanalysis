// Fits an affine transform to matched pairs between the micrograph and a transformed copy,
// with least squares on all pairs and with RANSAC, and checks how often RANSAC succeeds;
// pictures in out/feature-alignment/.
//   cargo run -p feature-alignment                      (your code)
//   cargo run -p feature-alignment --features solution  (the model solution)

use feature_alignment::*;
use imgcourse::*;

/// The true transform from view 1 to view 2: turned by about 10°, zoomed 1.1, a little shear, moved.
const TRUTH: Affine = [[1.06, 0.22, -20.0], [-0.17, 1.10, 6.0]];

/// Matched pairs: `good` correct ones (true transform plus 0.5 px noise), and `bad` wrong ones
/// (a random point of view 2), mixed in random order. Returns the pairs and which are correct.
fn make_pairs(good: usize, bad: usize, seed: u64) -> (Vec<Pair>, Vec<bool>) {
    let mut rng = Rng::new(seed);
    // Which pairs are correct, in random order (shuffle).
    let mut correct: Vec<bool> = (0..good + bad).map(|k| k < good).collect();
    for k in (1..correct.len()).rev() {
        correct.swap(k, rng.below(k + 1));
    }
    let mut pairs = Vec::new();
    for &is_good in &correct {
        // A point of view 1 whose image lies inside view 2.
        let (p, (u, v)) = loop {
            let p = (256.0 * rng.uniform(), 256.0 * rng.uniform());
            let q = apply(&TRUTH, p);
            if (5.0..251.0).contains(&q.0) && (5.0..251.0).contains(&q.1) {
                break (p, q);
            }
        };
        let q = if is_good { (u + 0.5 * rng.normal(), v + 0.5 * rng.normal()) } else { (256.0 * rng.uniform(), 256.0 * rng.uniform()) };
        pairs.push((p, q));
    }
    (pairs, correct)
}

/// The mean distance between where `m` and the true transform put the pixels of view 1.
fn true_error(m: &Affine) -> f32 {
    let mut sum = 0.0;
    for y in (0..256).step_by(8) {
        for x in (0..256).step_by(8) {
            let p = (x as f32, y as f32);
            let (a, b) = (apply(m, p), apply(&TRUTH, p));
            sum += (a.0 - b.0).hypot(a.1 - b.1);
        }
    }
    sum / 1024.0
}

/// A checkerboard of view 1 and view 2 read back through `m` (squares of 32 px): where the
/// fit is right, the structures continue across the squares.
fn checkerboard(view1: &Image<f32>, view2: &Image<f32>, m: &Affine) -> Image<f32> {
    view1.map_xy(|x, y, v| {
        if (x / 32 + y / 32) % 2 == 0 {
            v
        } else {
            let (u, w) = apply(m, (x as f32, y as f32));
            bilinear(view2, u, w)
        }
    })
}

fn main() {
    let dir = "out/feature-alignment";
    let view1 = samples::cell();
    // View 2: sample view 1 at the inverse transform of every pixel.
    let inverse = invert(&TRUTH);
    let view2 = view1.map_xy(|x, y, _| {
        let (u, v) = apply(&inverse, (x as f32, y as f32));
        bilinear(&view1, u, v)
    });
    save_gray(&view2, format!("{dir}/view2.png"));

    let (pairs, correct) = make_pairs(40, 60, 1);
    let w = 40.0 / 100.0;
    println!("{} pairs, 40 correct (w = {w})", pairs.len());
    let ls = fit_affine(&pairs).unwrap();
    println!("least squares on all pairs: true error {:.2} px", true_error(&ls));
    let good: Vec<Pair> = pairs.iter().zip(&correct).filter(|(_, &c)| c).map(|(p, _)| *p).collect();
    println!("least squares on the correct pairs only: true error {:.2} px", true_error(&fit_affine(&good).unwrap()));
    let n = ransac_iterations(w, 3, 0.99);
    println!("RANSAC iterations needed for w = {w}, s = 3, p = 0.99: {n}");
    let (m, found) = ransac_affine(&pairs, 2.0, n, &mut Rng::new(1)).unwrap();
    let wrong_in = found.iter().filter(|&&i| !correct[i]).count();
    println!("RANSAC ({n} iterations, ε = 2 px): {} inliers ({wrong_in} of them wrong pairs), true error {:.2} px", found.len(), true_error(&m));
    save_gray(&checkerboard(&view1, &view2, &ls), format!("{dir}/checker-least-squares.png"));
    save_gray(&checkerboard(&view1, &view2, &m), format!("{dir}/checker-ransac.png"));

    // How often does RANSAC find the right transform (true error below 1 px)? 200 runs each.
    println!("\nsuccess of RANSAC in 200 runs (true error < 1 px), against the prediction 1 − (1 − w³)^N:");
    for iterations in [5, 10, 20, n] {
        let mut ok = 0;
        for seed in 0..200 {
            if let Some((m, _)) = ransac_affine(&pairs, 2.0, iterations, &mut Rng::new(seed + 100)) {
                ok += (true_error(&m) < 1.0) as usize;
            }
        }
        let predicted = 1.0 - (1.0 - w * w * w).powi(iterations as i32);
        println!("  N = {iterations:3}: {:5.1} % (predicted {:5.1} %)", 100.0 * ok as f32 / 200.0, 100.0 * predicted);
    }
    println!("wrote the pictures to {dir}/");
}

fn invert(m: &Affine) -> Affine {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let (a, b, c, d) = (m[1][1] / det, -m[0][1] / det, -m[1][0] / det, m[0][0] / det);
    [[a, b, -(a * m[0][2] + b * m[1][2])], [c, d, -(c * m[0][2] + d * m[1][2])]]
}

fn bilinear(img: &Image<f32>, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let p = |i: f32, j: f32| img.get_border(i as isize, j as isize, Border::Zero);
    (1.0 - fy) * ((1.0 - fx) * p(x0, y0) + fx * p(x0 + 1.0, y0)) + fy * ((1.0 - fx) * p(x0, y0 + 1.0) + fx * p(x0 + 1.0, y0 + 1.0))
}
