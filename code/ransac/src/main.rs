// RANSAC against least squares on lines and circles with outliers, and how often RANSAC
// succeeds for a given number of samples. Pictures in out/ransac/.
//   cargo run -p ransac                      (your code)
//   cargo run -p ransac --features solution  (the model solution)

use imgcourse::*;
use ransac::sim::{canvas, circle_points, curve, dot, line_points};
use ransac::*;

/// RANSAC for a circle, the same loop as for a line: three points, `fit_circle` through them,
/// count the points within `eps` of the circle; refit on the inliers of the best.
fn ransac_circle(points: &[[f32; 2]], eps: f32, iterations: usize, rng: &mut Rng) -> ((f32, f32, f32), Vec<usize>) {
    let mut best: Vec<usize> = Vec::new();
    for _ in 0..iterations {
        let s = [rng.below(points.len()), rng.below(points.len()), rng.below(points.len())];
        if s[0] == s[1] || s[1] == s[2] || s[0] == s[2] {
            continue;
        }
        if let Some((a, b, r)) = fit_circle(&s.map(|k| points[k])) {
            let inliers: Vec<usize> = (0..points.len()).filter(|&k| ((points[k][0] - a).hypot(points[k][1] - b) - r).abs() <= eps).collect();
            if inliers.len() > best.len() {
                best = inliers;
            }
        }
    }
    let chosen: Vec<[f32; 2]> = best.iter().map(|&k| points[k]).collect();
    (fit_circle(&chosen).unwrap_or((0.0, 0.0, 0.0)), best)
}

/// Draw a line in normal form across the square.
fn draw_line(img: &mut Image<[f32; 3]>, l: &Line, size: f32, scale: f32, c: [f32; 3], dashed: bool) {
    let (n, d) = ([l.theta.cos(), l.theta.sin()], [-l.theta.sin(), l.theta.cos()]);
    curve(img, scale, c, dashed, |t| {
        let s = (t - 0.5) * 3.0 * size;
        [l.rho * n[0] + s * d[0], l.rho * n[1] + s * d[1]]
    });
}

fn main() {
    let dir = "out/ransac";
    let size = 100.0;
    let scale = 3.0;
    let (green, grey, blue, orange) = ([0.1, 0.7, 0.2], [0.6, 0.6, 0.6], [0.1, 0.3, 0.9], [1.0, 0.5, 0.0]);

    // 1. A line θ = 63°, ρ = 58 with 30 points (noise 0.8) and 30 outliers.
    let mut rng = Rng::new(1);
    let pts = line_points(size, 63.0, 58.0, 0.8, 30, 30, &mut rng);
    let ls = fit_line_tls(&pts);
    let n = iterations_needed(0.5, 2, 0.99);
    let (best, inliers) = ransac_line(&pts, 2.0, n, &mut rng);
    let true_in = inliers.iter().filter(|&&k| k < 30).count();
    println!("line θ = 63°, ρ = 58; 30 inliers, 30 outliers");
    println!("  least squares on all points: θ = {:.1}°, ρ = {:.1}", ls.theta.to_degrees(), ls.rho);
    println!("  RANSAC, {n} samples, ε = 2: θ = {:.1}°, ρ = {:.1}, {} inliers ({true_in} true)", best.theta.to_degrees(), best.rho, inliers.len());
    let mut img = canvas(size, scale);
    for (k, &p) in pts.iter().enumerate() {
        dot(&mut img, p, scale, if inliers.contains(&k) { green } else { grey });
    }
    draw_line(&mut img, &ls, size, scale, orange, true);
    draw_line(&mut img, &best, size, scale, blue, false);
    save_rgb(&img, format!("{dir}/line.png"));

    // 2. How often does RANSAC find the line? 1000 new data sets each; success = refit within
    // 2° and 2 px of the truth.
    println!("success rate over 1000 data sets (predicted: chance of at least one all-inlier sample)");
    for (n_in, n_out, iters) in [(30, 30, 17), (20, 80, 17), (20, 80, 113)] {
        let w = n_in as f32 / (n_in + n_out) as f32;
        let mut ok = 0;
        let mut rng = Rng::new(7);
        for _ in 0..1000 {
            let pts = line_points(size, 63.0, 58.0, 0.8, n_in, n_out, &mut rng);
            let (l, _) = ransac_line(&pts, 2.0, iters, &mut rng);
            if (l.theta.to_degrees() - 63.0).abs() < 2.0 && (l.rho - 58.0).abs() < 2.0 {
                ok += 1;
            }
        }
        let predicted = 1.0 - (1.0 - w * w).powi(iters as i32);
        println!("  w = {w:.1}, {iters:>3} samples: {:.1} % found, predicted {:.1} %", ok as f32 / 10.0, 100.0 * predicted);
    }

    // 3. A circle, centre (52, 48), r = 28, with 30 points and 30 outliers.
    let mut rng = Rng::new(2);
    let pts = circle_points(size, [52.0, 48.0], 28.0, 0.8, 30, 30, &mut rng);
    let (a, b, r) = fit_circle(&pts).expect("circle");
    let n = iterations_needed(0.5, 3, 0.99);
    let ((ra, rb, rr), inliers) = ransac_circle(&pts, 2.0, n, &mut rng);
    println!("circle (52, 48), r = 28; 30 inliers, 30 outliers");
    println!("  least squares on all points: ({a:.1}, {b:.1}), r = {r:.1}");
    println!("  RANSAC, {n} samples: ({ra:.1}, {rb:.1}), r = {rr:.1}, {} inliers", inliers.len());
    let mut img = canvas(size, scale);
    for (k, &p) in pts.iter().enumerate() {
        dot(&mut img, p, scale, if inliers.contains(&k) { green } else { grey });
    }
    let circle = |a: f32, b: f32, r: f32| move |t: f32| [a + r * (std::f32::consts::TAU * t).cos(), b + r * (std::f32::consts::TAU * t).sin()];
    curve(&mut img, scale, orange, true, circle(a, b, r));
    curve(&mut img, scale, blue, false, circle(ra, rb, rr));
    save_rgb(&img, format!("{dir}/circle.png"));
    println!("wrote the pictures to {dir}/ (blue = RANSAC, orange dashed = least squares, green = inliers)");
}
