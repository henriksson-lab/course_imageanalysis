// Finds DoG keypoints on the course micrograph and on zoomed and turned copies of it, and
// measures how many are found again (repeatability); pictures in out/keypoints/.
//   cargo run -p keypoints -- keypoints                      (your code)
//   cargo run -p keypoints --features solution -- keypoints  (the model solution)

use imgcourse::*;
use crate::common::bilinear;
use crate::keypoints::*;

const SIGMA0: f32 = 1.6;
const S: usize = 3;
const CONTRAST: f32 = 0.03; // |D| at the refined peak, grey values in [0, 1]
const EDGE_R: f32 = 10.0;

/// A keypoint: position, scale (in original pixels) and orientation (degrees, counter-clockwise on screen).
#[derive(Clone, Copy, Debug)]
struct Keypoint {
    x: f32,
    y: f32,
    sigma: f32,
    theta: f32,
}

#[derive(Default, Debug)]
struct Counts {
    candidates: usize,
    weak: usize,
    edge: usize,
    kept: usize,
}

/// The full detector of the lecture: octaves, extrema, refinement, contrast and edge tests,
/// orientations. With `edge_r = None` the edge test is skipped.
fn detect(img: &Image<f32>, octaves: usize, edge_r: Option<f32>, counts: &mut Counts) -> Vec<Keypoint> {
    let mut keypoints = Vec::new();
    // The image is taken to be blurred with σ = 0.5 already; bring it to σ0.
    let mut base = gaussian_blur(img, (SIGMA0 * SIGMA0 - 0.25).sqrt());
    for octave in 0..octaves {
        if base.width() < 16 {
            break;
        }
        let (blurred, dogs) = dog_octave(&base, SIGMA0, S);
        for (x, y, level) in local_extrema(&dogs, 0.5 * CONTRAST) {
            counts.candidates += 1;
            let d = |dl: isize, dx: isize, dy: isize| {
                dogs[(level as isize + dl) as usize].get_border(x as isize + dx, y as isize + dy, Border::Clamp)
            };
            // Refine along x, y and the level, one parabola each. (At an extremum each
            // offset is at most half a sample; the full 3D fit of the lecture can move further.)
            let (ox, vx) = parabola_peak(d(0, -1, 0), d(0, 0, 0), d(0, 1, 0));
            let (oy, vy) = parabola_peak(d(0, 0, -1), d(0, 0, 0), d(0, 0, 1));
            let (ol, vl) = parabola_peak(d(-1, 0, 0), d(0, 0, 0), d(1, 0, 0));
            let value = vx + vy + vl - 2.0 * d(0, 0, 0);
            if value.abs() < CONTRAST {
                counts.weak += 1;
                continue;
            }
            if let Some(r) = edge_r {
                if !edge_test(&dogs[level], x, y, r) {
                    counts.edge += 1;
                    continue;
                }
            }
            counts.kept += 1;
            let scale = 2f32.powi(octave as i32);
            let sigma_octave = level_sigma(SIGMA0, S, 0, level as f32 + ol); // in this octave's pixels
            let (px, py) = (x as f32 + ox, y as f32 + oy);
            for theta in orientations(&blurred[level], px, py, sigma_octave) {
                keypoints.push(Keypoint { x: px * scale, y: py * scale, sigma: sigma_octave * scale, theta });
            }
        }
        // The next octave starts from the image with σ = 2σ0, every second pixel.
        let b = &blurred[S];
        base = Image::from_fn(b.width() / 2, b.height() / 2, |x, y| b[(2 * x, 2 * y)]);
    }
    keypoints
}

/// The gradients in a disc around the keypoint, weighted by a Gaussian of 1.5σ, as votes
/// for the orientation histogram.
fn orientations(l: &Image<f32>, px: f32, py: f32, sigma: f32) -> Vec<f32> {
    let w = 1.5 * sigma;
    let r = (3.0 * w).round() as isize;
    let mut votes = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            let (x, y) = (px.round() as isize + dx, py.round() as isize + dy);
            if dx * dx + dy * dy > r * r || !l.contains(x, y) {
                continue;
            }
            let gx = (l.get_border(x + 1, y, Border::Clamp) - l.get_border(x - 1, y, Border::Clamp)) / 2.0;
            let gy = (l.get_border(x, y + 1, Border::Clamp) - l.get_border(x, y - 1, Border::Clamp)) / 2.0;
            let weight = (-((dx * dx + dy * dy) as f32) / (2.0 * w * w)).exp();
            // y points down: counter-clockwise on the screen is atan2(−gy, gx).
            votes.push(((gx * gx + gy * gy).sqrt() * weight, (-gy).atan2(gx).to_degrees()));
        }
    }
    dominant_orientations(&orientation_histogram(&votes))
}

/// Keypoints at one fixed scale: 2D extrema of the DoG at σ ≈ 2 (level 1 of octave 0),
/// without refinement, as the "one σ" detector of the lecture.
fn detect_one_scale(img: &Image<f32>) -> Vec<Keypoint> {
    let base = gaussian_blur(img, (SIGMA0 * SIGMA0 - 0.25).sqrt());
    let (_, dogs) = dog_octave(&base, SIGMA0, S);
    let d = &dogs[1];
    let mut points = Vec::new();
    for y in 1..d.height() - 1 {
        for x in 1..d.width() - 1 {
            let v = d[(x, y)];
            let mut is_max = v.abs() > CONTRAST;
            for (dx, dy) in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                let n = d.get_border(x as isize + dx, y as isize + dy, Border::Clamp);
                is_max &= v.abs() > n.abs();
            }
            if is_max {
                points.push(Keypoint { x: x as f32, y: y as f32, sigma: level_sigma(SIGMA0, S, 0, 1.0), theta: 0.0 });
            }
        }
    }
    points
}

/// View 2: the image zoomed by `zoom` and turned by `angle` degrees about its centre
/// (bilinear interpolation, 0 outside). Returns the image and the map view 1 → view 2.
fn zoom_and_turn(img: &Image<f32>, zoom: f32, angle: f32) -> (Image<f32>, impl Fn(f32, f32) -> (f32, f32)) {
    let (cx, cy) = (img.width() as f32 / 2.0, img.height() as f32 / 2.0);
    let (s, c) = angle.to_radians().sin_cos();
    let forward = move |x: f32, y: f32| {
        let (dx, dy) = (x - cx, y - cy);
        (cx + zoom * (c * dx - s * dy), cy + zoom * (s * dx + c * dy))
    };
    let out = Image::from_fn(img.width(), img.height(), |x, y| {
        // Inverse map: where in view 1 does this pixel of view 2 come from?
        let (dx, dy) = ((x as f32 - cx) / zoom, (y as f32 - cy) / zoom);
        bilinear(img, cx + c * dx + s * dy, cy - s * dx + c * dy)
    });
    (out, forward)
}

/// Repeatability: of the view-1 points whose image under `t` lies well inside view 2, the
/// fraction with a view-2 point within `eps` px and (if `check_scale`) a scale within a
/// factor 1.5 of the expected one; each view-2 point used once. Returns (found, possible).
fn repeatability(p1: &[Keypoint], p2: &[Keypoint], t: &impl Fn(f32, f32) -> (f32, f32), zoom: f32, size: f32, check_scale: bool) -> (usize, usize) {
    let eps = 2.5;
    let mut used = vec![false; p2.len()];
    let (mut found, mut possible) = (0, 0);
    // Count each position once (a keypoint with two orientations is one point).
    let mut seen: Vec<(f32, f32)> = Vec::new();
    for p in p1 {
        if seen.iter().any(|&(x, y)| (x - p.x).abs() < 0.01 && (y - p.y).abs() < 0.01) {
            continue;
        }
        seen.push((p.x, p.y));
        let (x, y) = t(p.x, p.y);
        if x < 10.0 || y < 10.0 || x > size - 10.0 || y > size - 10.0 {
            continue;
        }
        possible += 1;
        let hit = p2.iter().enumerate().position(|(j, q)| {
            !used[j] && (q.x - x).hypot(q.y - y) < eps && (!check_scale || (zoom * p.sigma / q.sigma).ln().abs() < 1.5f32.ln())
        });
        if let Some(j) = hit {
            used[j] = true;
            found += 1;
        }
    }
    (found, possible)
}

/// Draw the keypoints as circles of radius 2σ with a line towards θ, on a copy enlarged twice.
fn draw(img: &Image<f32>, points: &[Keypoint]) -> Image<[f32; 3]> {
    let big = Image::from_fn(2 * img.width(), 2 * img.height(), |x, y| img[(x / 2, y / 2)]);
    let mut out = gray_to_rgb(&big);
    let mut put = |x: f32, y: f32, colour: [f32; 3]| {
        let (x, y) = ((2.0 * x).round() as isize, (2.0 * y).round() as isize);
        if out.contains(x, y) {
            out[(x as usize, y as usize)] = colour;
        }
    };
    for p in points {
        let r = 2.0 * p.sigma;
        let steps = (2.0 * std::f32::consts::PI * 2.0 * r) as usize + 8;
        for i in 0..steps {
            let a = i as f32 / steps as f32 * std::f32::consts::TAU;
            put(p.x + r * a.cos(), p.y + r * a.sin(), [1.0, 0.85, 0.0]);
        }
        let (s, c) = p.theta.to_radians().sin_cos();
        for i in 0..(4.0 * r) as usize {
            let t = i as f32 / 4.0;
            put(p.x + t * c, p.y - t * s, [0.0, 1.0, 0.3]);
        }
    }
    out
}

/// A synthetic scene: bright and dark discs of radius 3, 6 and 12 on grey, and a long thin
/// bar whose brightness rises and falls along it.
fn scene() -> Image<f32> {
    let discs = [(30.0, 32.0, 3.0, 0.9), (96.0, 92.0, 3.0, 0.1), (60.0, 140.0, 6.0, 0.9), (150.0, 60.0, 6.0, 0.1), (140.0, 150.0, 12.0, 0.85), (40.0, 80.0, 12.0, 0.15)];
    let img = Image::from_fn(192, 192, |x, y| {
        let (x, y) = (x as f32, y as f32);
        let mut v = 0.5;
        for (cx, cy, r, b) in discs {
            if (x - cx).hypot(y - cy) <= r {
                v = b;
            }
        }
        if (y - 175.0).abs() <= 2.0 && (15.0..=177.0).contains(&x) {
            v = 0.8 + 0.2 * (x / 10.0).sin();
        }
        v
    });
    gaussian_blur(&img, 0.7)
}

pub fn main() {
    let dir = "out/keypoints";
    let img = samples::cell();
    let size = img.width() as f32;

    let mut counts = Counts::default();
    let points = detect(&img, 5, Some(EDGE_R), &mut counts);
    println!("micrograph: {} candidates, {} too weak, {} on an edge, {} kept ({} keypoints with orientations)", counts.candidates, counts.weak, counts.edge, counts.kept, points.len());
    save_rgb(&draw(&img, &points), format!("{dir}/keypoints.png"));
    let mut no_edge = Counts::default();
    let all = detect(&img, 5, None, &mut no_edge);
    println!("without the edge test: {} kept ({} keypoints)", no_edge.kept, all.len());
    save_rgb(&draw(&img, &all), format!("{dir}/keypoints-no-edge-test.png"));

    // The edge test on the synthetic scene with a long bar.
    let sc = scene();
    let mut c1 = Counts::default();
    let with = detect(&sc, 5, Some(EDGE_R), &mut c1);
    let mut c2 = Counts::default();
    let without = detect(&sc, 5, None, &mut c2);
    println!("synthetic scene: {} kept with the edge test ({} removed as edges), {} kept without it", c1.kept, c1.edge, c2.kept);
    save_rgb(&draw(&sc, &with), format!("{dir}/scene-edge-test.png"));
    save_rgb(&draw(&sc, &without), format!("{dir}/scene-no-edge-test.png"));

    println!("\nrepeatability (found again / possible), view 2 = view 1 zoomed and turned by 20°:");
    println!("  zoom   one σ            DoG");
    for zoom in [1.0, 1.5, 2.0, 2.5] {
        let (view2, t) = zoom_and_turn(&img, zoom, 20.0);
        let one = repeatability(&detect_one_scale(&img), &detect_one_scale(&view2), &t, zoom, size, false);
        let dog = repeatability(&points, &detect(&view2, 5, Some(EDGE_R), &mut Counts::default()), &t, zoom, size, true);
        let pct = |(f, p): (usize, usize)| 100.0 * f as f32 / p.max(1) as f32;
        println!("  {zoom:.1}    {:3} / {:3} {:3.0} %   {:3} / {:3} {:3.0} %", one.0, one.1, pct(one), dog.0, dog.1, pct(dog));
        if zoom == 2.0 {
            save_gray(&view2, format!("{dir}/view2-zoom2.png"));
            save_rgb(&draw(&view2, &detect(&view2, 5, Some(EDGE_R), &mut Counts::default())), format!("{dir}/keypoints-zoom2.png"));
        }
    }
    println!("wrote the pictures to {dir}/");
}
