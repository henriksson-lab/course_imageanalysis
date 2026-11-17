// Bundle adjustment in flatland with your code: Gauss–Newton against Levenberg–Marquardt, and
// squared against Huber loss with wrong observations. Pictures in out/sfm/.
//   cargo run -p sfm                      (your code)
//   cargo run -p sfm --features solution  (the model solution)

use imgcourse::*;
use sfm::*;

/// 10 cameras on a circle of radius 4 around the object, each turned slightly off the centre.
/// Camera 1 stands opposite camera 0, so that fixing its x (the gauge) fixes the scale well.
fn true_cameras() -> Vec<Cam> {
    [0, 5, 1, 2, 3, 4, 6, 7, 8, 9]
        .iter()
        .map(|&j| {
            let a = 2.0 * std::f64::consts::PI * j as f64 / 10.0;
            [4.0 * a.cos(), 4.0 * a.sin(), a + std::f64::consts::PI + 0.05 * (j as f64 - 4.5) / 4.5]
        })
        .collect()
}

/// 40 points on a wavy outline around the origin.
fn true_points() -> Vec<Pt> {
    (0..40)
        .map(|i| {
            let a = 2.0 * std::f64::consts::PI * i as f64 / 40.0;
            let r = 1.2 + 0.35 * (3.0 * a).sin();
            [r * a.cos(), r * a.sin()]
        })
        .collect()
}

/// Every camera sees the points on its side of the outline that fall inside its 160 pixels:
/// the near half (a point is hidden if the outline's centre is closer to the camera than it).
fn observe(cams: &[Cam], points: &[Pt], sigma: f64, rng: &mut Rng) -> Vec<Obs> {
    let mut obs = Vec::new();
    for (j, c) in cams.iter().enumerate() {
        for (i, p) in points.iter().enumerate() {
            let to_cam = [c[0] - p[0], c[1] - p[1]];
            let facing = p[0] * to_cam[0] + p[1] * to_cam[1] > 0.0;
            let u = project(c, p);
            if facing && (0.0..160.0).contains(&u) {
                obs.push(Obs { cam: j, point: i, u: u + sigma * rng.normal() as f64 });
            }
        }
    }
    obs
}

/// Move every free parameter by Gaussian noise of standard deviation `size` (radians are
/// scaled down by 5).
fn perturb(cams: &[Cam], points: &[Pt], size: f64, rng: &mut Rng) -> (Vec<Cam>, Vec<Pt>) {
    let mut theta = pack(cams, points);
    for (n, v) in theta.iter_mut().enumerate().skip(FIXED) {
        let angle = n < 3 * cams.len() && n % 3 == 2;
        *v += size * rng.normal() as f64 * if angle { 0.2 } else { 1.0 };
    }
    let (mut c, mut p) = (cams.to_vec(), points.to_vec());
    unpack(&theta, &mut c, &mut p);
    (c, p)
}

/// Draw the scene from above: true points grey, estimated points orange, cameras blue (a dot
/// and a short line along the viewing direction). 300 × 300 pixels, 1 unit = 30 pixels.
fn draw(cams_true: &[Cam], points_true: &[Pt], cams: &[Cam], points: &[Pt]) -> Image<[f32; 3]> {
    let mut img = Image::new(300, 300, [1.0f32; 3]);
    let dot = |p: [f64; 2], r: isize, colour: [f32; 3], img: &mut Image<[f32; 3]>| {
        let (x, y) = (150.0 + 30.0 * p[0], 150.0 - 30.0 * p[1]);
        for dy in -r..=r {
            for dx in -r..=r {
                let (px, py) = (x as isize + dx, y as isize + dy);
                if img.contains(px, py) {
                    img[(px as usize, py as usize)] = colour;
                }
            }
        }
    };
    for p in points_true {
        dot(*p, 2, [0.7, 0.7, 0.7], &mut img);
    }
    for p in points {
        dot(*p, 1, [0.95, 0.5, 0.0], &mut img);
    }
    for c in cams_true.iter().chain(cams) {
        for k in 0..12 {
            let t = k as f64 * 0.05;
            dot([c[0] + t * c[2].cos(), c[1] + t * c[2].sin()], 0, [0.1, 0.3, 0.9], &mut img);
        }
        dot([c[0], c[1]], 2, [0.1, 0.3, 0.9], &mut img);
    }
    img
}

fn show(h: &[f64]) -> String {
    h.iter().map(|v| if v.is_finite() && *v < 1e5 { format!("{v:.2}") } else { "huge".to_string() }).collect::<Vec<_>>().join(" → ")
}

fn main() {
    let dir = "out/sfm";
    let mut rng = Rng::new(2);
    let (cams, points) = (true_cameras(), true_points());
    let obs = observe(&cams, &points, 0.2, &mut rng);
    println!("{} cameras, {} points, {} observations with 0.2 px noise; {} unknowns", cams.len(), points.len(), obs.len(), pack(&cams, &points).len() - FIXED);

    // 1. Gauss–Newton and Levenberg–Marquardt, from a small and from a large perturbation.
    for (name, size, iterations) in [("small", 0.05, 6), ("large", 0.6, 15)] {
        let (c0, p0) = perturb(&cams, &points, size, &mut rng);
        for lm in [false, true] {
            let (mut c, mut p) = (c0.clone(), p0.clone());
            let h = bundle_adjust(&mut c, &mut p, &obs, iterations, lm, None);
            println!("{name} perturbation, {}: RMS {} px", if lm { "Levenberg–Marquardt" } else { "Gauss–Newton       " }, show(&h));
            if name == "large" && lm {
                save_rgb(&draw(&cams, &points, &c0, &p0), format!("{dir}/ba-start.png"));
                save_rgb(&draw(&cams, &points, &c, &p), format!("{dir}/ba-lm.png"));
            }
        }
    }

    // 2. Three wrong observations, moved by 20 to 28 pixels: squared loss against Huber.
    let mut bad_obs = obs.clone();
    let wrong = [5, 50, 120];
    for (k, &i) in wrong.iter().enumerate() {
        bad_obs[i].u += [20.0, -24.0, 28.0][k];
    }
    let (c0, p0) = perturb(&cams, &points, 0.05, &mut rng);
    for delta in [None, Some(1.0)] {
        let (mut c, mut p) = (c0.clone(), p0.clone());
        bundle_adjust(&mut c, &mut p, &bad_obs, 20, true, delta);
        let r = residuals(&c, &p, &bad_obs);
        let good: Vec<f64> = (0..r.len()).filter(|i| !wrong.contains(i)).map(|i| r[i]).collect();
        let err = points.iter().zip(&p).map(|(a, b)| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()).fold(0.0, f64::max);
        println!(
            "3 wrong observations, {}: correct ones RMS {:.2} px, wrong ones {:.1} {:.1} {:.1} px, largest point error {:.3}",
            if delta.is_some() { "Huber δ = 1  " } else { "squared loss " },
            rms(&good),
            r[wrong[0]],
            r[wrong[1]],
            r[wrong[2]],
            err
        );
    }
    println!("wrote the pictures to {dir}/");
}
