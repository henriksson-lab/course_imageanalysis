// Estimates the fundamental matrix of a synthetic two-camera scene with outliers, by least
// squares and by RANSAC, draws epipolar lines, and triangulates the matches. Pictures in
// out/two-view/.
//   cargo run -p two-view                      (your code)
//   cargo run -p two-view --features solution  (the model solution)

use imgcourse::*;
use two_view::*;

const W: usize = 640;
const H: usize = 480;
const FOCAL: f64 = 700.0;

fn project(x: Vec3) -> [f64; 2] {
    [FOCAL * x[0] / x[2] + 320.0, FOCAL * x[1] / x[2] + 240.0]
}

/// Draw the part of the line a u + b v + c = 0 that lies in the image.
fn draw_line(img: &mut Image<[f32; 3]>, l: Vec3, colour: [f32; 3]) {
    if l[1].abs() > l[0].abs() {
        for u in 0..W {
            let v = -(l[0] * u as f64 + l[2]) / l[1];
            if v >= 0.0 && v < H as f64 {
                img[(u, v as usize)] = colour;
            }
        }
    } else {
        for v in 0..H {
            let u = -(l[1] * v as f64 + l[2]) / l[0];
            if u >= 0.0 && u < W as f64 {
                img[(u as usize, v)] = colour;
            }
        }
    }
}

fn dot_at(img: &mut Image<[f32; 3]>, p: [f64; 2], colour: [f32; 3]) {
    for dy in -2..=2isize {
        for dx in -2..=2isize {
            let (x, y) = (p[0] as isize + dx, p[1] as isize + dy);
            if img.contains(x, y) && dx.abs() + dy.abs() <= 2 {
                img[(x as usize, y as usize)] = colour;
            }
        }
    }
}

fn main() {
    let dir = "out/two-view";
    let mut rng = Rng::new(7);

    // The scene: 80 points 4–9 m in front of camera 1. Camera 2 stands 1.2 m to the right and
    // 0.3 m forward, turned 10° towards the scene: X2 = R X1 + t.
    let (s, c) = (-10f64).to_radians().sin_cos();
    let r = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]];
    let centre2 = [1.2, 0.0, 0.3];
    let rc = mat_vec(&r, centre2);
    let t = [-rc[0], -rc[1], -rc[2]];
    let mut points = Vec::new();
    let mut matches = Vec::new();
    let sigma = 0.5;
    while points.len() < 80 {
        let x = [6.0 * rng.uniform() as f64 - 2.5, 3.0 * rng.uniform() as f64 - 1.5, 4.0 + 5.0 * rng.uniform() as f64];
        let x2 = mat_vec(&r, x);
        let (p1, p2) = (project(x), project([x2[0] + t[0], x2[1] + t[1], x2[2] + t[2]]));
        let inside = |p: [f64; 2]| p[0] >= 0.0 && p[1] >= 0.0 && p[0] < W as f64 && p[1] < H as f64;
        if inside(p1) && inside(p2) {
            points.push(x);
            let noise = |p: [f64; 2], rng: &mut Rng| [p[0] + sigma * rng.normal() as f64, p[1] + sigma * rng.normal() as f64];
            matches.push((noise(p1, &mut rng), noise(p2, &mut rng)));
        }
    }
    // a quarter of the matches are wrong: their pixel in image 2 is anywhere
    let n_wrong = 20;
    for m in matches.iter_mut().take(n_wrong) {
        m.1 = [W as f64 * rng.uniform() as f64, H as f64 * rng.uniform() as f64];
    }
    let correct: Vec<usize> = (n_wrong..matches.len()).collect();
    println!("{} matches, {n_wrong} of them wrong, noise {sigma} px", matches.len());

    // 1. Least squares on all matches, and RANSAC.
    let f_ls = eight_point(&matches);
    let iterations = ransac_iterations(0.99, 0.5, 8);
    let (f_ransac, inliers) = ransac_fundamental(&matches, 1.5, iterations, &mut rng);
    let mean_dist = |f: &Mat3| correct.iter().map(|&k| line_distance(epipolar_line(f, matches[k].0), matches[k].1)).sum::<f64>() / correct.len() as f64;
    println!("eight-point on all matches: the correct matches lie {:.2} px from their epipolar lines", mean_dist(&f_ls));
    let found_correct = inliers.iter().filter(|&&k| k >= n_wrong).count();
    println!("RANSAC ({iterations} samples, ε = 1.5 px): {} inliers, {found_correct} of them correct; the correct matches lie {:.2} px from their lines", inliers.len(), mean_dist(&f_ransac));

    // 2. Pictures of image 2: the points (black: correct matches, red: wrong ones) and the
    //    epipolar lines of eight points of image 1.
    for (name, f) in [("ls", &f_ls), ("ransac", &f_ransac)] {
        let mut img = Image::new(W, H, [1.0f32; 3]);
        for k in (n_wrong..matches.len()).step_by(8) {
            draw_line(&mut img, epipolar_line(f, matches[k].0), [0.2, 0.6, 0.2]);
        }
        for (k, m) in matches.iter().enumerate() {
            dot_at(&mut img, m.1, if k < n_wrong { [0.9, 0.1, 0.1] } else { [0.0, 0.0, 0.0] });
        }
        save_rgb(&img, format!("{dir}/epipolar-{name}.png"));
    }

    // 3. Triangulation with the known cameras: the depth error against the depth.
    let rt = [0, 1, 2].map(|i| [0, 1, 2].map(|j| r[j][i])); // Rᵀ
    let ray1 = |p: [f64; 2]| [(p[0] - 320.0) / FOCAL, (p[1] - 240.0) / FOCAL, 1.0];
    let mut near = (0.0, 0, 0.0);
    let mut far = (0.0, 0, 0.0);
    for &k in &correct {
        let d2 = mat_vec(&rt, ray1(matches[k].1)); // camera 2's ray in world coordinates
        let x = triangulate_midpoint([0.0; 3], ray1(matches[k].0), centre2, d2);
        let err = (x[2] - points[k][2]).powi(2);
        if points[k][2] < 6.0 {
            near = (near.0 + err, near.1 + 1, near.2 + points[k][2]);
        } else if points[k][2] > 7.5 {
            far = (far.0 + err, far.1 + 1, far.2 + points[k][2]);
        }
    }
    for (name, g) in [("nearer than 6 m", near), ("beyond 7.5 m", far)] {
        println!("triangulation, {} points {name} (mean depth {:.2} m): RMS depth error {:.3} m", g.1, g.2 / g.1 as f64, (g.0 / g.1 as f64).sqrt());
    }
    println!("wrote the pictures to {dir}/");
}
