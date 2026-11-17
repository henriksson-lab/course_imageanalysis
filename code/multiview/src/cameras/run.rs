// Calibrates a hidden camera from a synthetic two-wall checkerboard target with your DLT, and
// undistorts a checkerboard image. Pictures in out/cameras/.
//   cargo run -p multiview -- cameras                      (your code)
//   cargo run -p multiview --features solution -- cameras  (the model solution)

use crate::cameras::*;
use imgcourse::*;

const W: usize = 640;
const H: usize = 480;

/// A rotation by `deg` degrees about the optical axis (z).
fn roll(deg: f64) -> Mat3 {
    let (s, c) = deg.to_radians().sin_cos();
    [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
}

/// The hidden camera: f = 820 px, principal point (335, 245), looking at the corner of the
/// two walls from the front, rolled by 4°.
fn hidden() -> Camera {
    let (r, t) = look_at([0.55, -0.3, 0.6], [0.12, -0.05, 0.12], [0.0, 1.0, 0.0]);
    Camera { f: 820.0, cx: 335.0, cy: 245.0, r: mat_mul(&roll(4.0), &r), t: mat_vec(&roll(4.0), t) }
}

/// 72 corners on two checkerboard walls at right angles (z = 0 and x = 0), 5 cm apart.
fn walls() -> Vec<Vec3> {
    let mut pts = Vec::new();
    for i in 1..=6 {
        for j in 1..=6 {
            pts.push([0.05 * i as f64, 0.05 * j as f64 - 0.2, 0.0]);
            pts.push([0.0, 0.05 * j as f64 - 0.2, 0.05 * i as f64]);
        }
    }
    pts
}

/// What the camera records: the projection, with radial distortion k1 and Gaussian pixel
/// noise of standard deviation `sigma`.
fn observe(cam: &Camera, pts: &[Vec3], k1: f64, sigma: f64, rng: &mut Rng) -> Vec<[f64; 2]> {
    pts.iter()
        .map(|&x| {
            let c = mat_vec(&cam.r, x);
            let c = [c[0] + cam.t[0], c[1] + cam.t[1], c[2] + cam.t[2]];
            let d = distort(c[0] / c[2], c[1] / c[2], k1, 0.0);
            [cam.f * d[0] + cam.cx + sigma * rng.normal() as f64, cam.f * d[1] + cam.cy + sigma * rng.normal() as f64]
        })
        .collect()
}

fn draw_line(img: &mut Image<[f32; 3]>, a: [f64; 2], b: [f64; 2], colour: [f32; 3]) {
    let n = ((b[0] - a[0]).abs().max((b[1] - a[1]).abs()) * 2.0).ceil().max(1.0) as usize;
    for i in 0..=n {
        let t = i as f64 / n as f64;
        let (x, y) = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]));
        if x >= 0.0 && y >= 0.0 && (x as usize) < W && (y as usize) < H {
            img[(x as usize, y as usize)] = colour;
        }
    }
}

/// Calibrate from the given points, print the result, and draw the residual arrows (30×).
fn calibrate(name: &str, pts: &[Vec3], seen: &[[f64; 2]], picture: Option<&str>) {
    let p = dlt(pts, seen);
    let (k, _, _) = decompose(&p);
    let mut sum = 0.0;
    let mut img = Image::new(W, H, [1.0f32; 3]);
    for (x, s) in pts.iter().zip(seen) {
        let pred = project_p(&p, *x);
        sum += (pred[0] - s[0]).powi(2) + (pred[1] - s[1]).powi(2);
        let tip = [s[0] + 30.0 * (pred[0] - s[0]), s[1] + 30.0 * (pred[1] - s[1])];
        draw_line(&mut img, *s, tip, [0.9, 0.1, 0.1]);
        for (dx, dy) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)] {
            draw_line(&mut img, [s[0] + dx, s[1] + dy], [s[0] + dx, s[1] + dy], [0.0, 0.0, 0.0]);
        }
    }
    let rms = (sum / pts.len() as f64).sqrt();
    println!("{name:38}: fx = {:7.1}, fy = {:7.1}, principal point ({:6.1}, {:6.1}), skew {:6.2}, RMS {rms:.2} px", k[0][0], k[1][1], k[0][2], k[1][2], k[0][1]);
    if let Some(file) = picture {
        save_rgb(&img, file);
    }
}

pub fn main() {
    let dir = "out/cameras";
    let cam = hidden();
    let pts = walls();
    let mut rng = Rng::new(3);
    println!("the hidden camera: f = 820, principal point (335, 245)\n");

    // 1. Calibration with the DLT.
    let six = [0, 11, 61, 70, 33, 40];
    let (six_pts, six_px): (Vec<Vec3>, Vec<[f64; 2]>) = {
        let exact = observe(&cam, &pts, 0.0, 0.0, &mut rng);
        six.iter().map(|&i| (pts[i], exact[i])).unzip()
    };
    calibrate("6 points, no noise", &six_pts, &six_px, None);
    let noisy = observe(&cam, &pts, 0.0, 0.5, &mut rng);
    let six_noisy: Vec<[f64; 2]> = six.iter().map(|&i| noisy[i]).collect();
    calibrate("6 points, noise 0.5 px", &six_pts, &six_noisy, None);
    calibrate("72 points, noise 0.5 px", &pts, &noisy, Some(&format!("{dir}/residuals-noise.png")));
    let barrel = observe(&cam, &pts, -0.3, 0.1, &mut rng);
    calibrate("72 points, k1 = -0.3, noise 0.1 px", &pts, &barrel, Some(&format!("{dir}/residuals-distortion.png")));
    // all points on one wall: the DLT has no unique answer
    let one_wall: Vec<usize> = (0..pts.len()).filter(|&i| pts[i][2] == 0.0).collect();
    let (wall_pts, wall_px): (Vec<Vec3>, Vec<[f64; 2]>) = one_wall.iter().map(|&i| (pts[i], noisy[i])).unzip();
    calibrate("36 points on one wall, noise 0.5 px", &wall_pts, &wall_px, None);

    // 2. Radial distortion of a checkerboard image (k1 = −0.3), and undistortion back.
    //    Both are inverse mappings: for every output pixel, ask where it comes from.
    let (f, cx, cy, k1) = (820.0, 320.0, 240.0, -0.3);
    let ideal = samples::checker(W, H, 40);
    let sample = |img: &Image<f32>, u: f64, v: f64| img.get_border(u.round() as isize, v.round() as isize, Border::Zero);
    let distorted = Image::from_fn(W, H, |u, v| {
        let p = undistort((u as f64 - cx) / f, (v as f64 - cy) / f, k1, 0.0);
        sample(&ideal, f * p[0] + cx, f * p[1] + cy)
    });
    let restored = Image::from_fn(W, H, |u, v| {
        let p = distort((u as f64 - cx) / f, (v as f64 - cy) / f, k1, 0.0);
        sample(&distorted, f * p[0] + cx, f * p[1] + cy)
    });
    save_gray(&distorted, format!("{dir}/checker-barrel.png"));
    save_gray(&restored, format!("{dir}/checker-undistorted.png"));
    let corner = distort((0.0 - cx) / f, (0.0 - cy) / f, k1, 0.0);
    println!("\nwith k1 = {k1}, the image corner (0, 0) is seen at ({:.1}, {:.1})", f * corner[0] + cx, f * corner[1] + cy);
    println!("wrote the pictures to {dir}/");
}
