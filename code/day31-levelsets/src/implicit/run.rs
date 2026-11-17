// Uses your set operations, crossings and curvature on curves stored as images φ;
// pictures in out/implicit/.
//   cargo run -p day31 -- implicit                      (your code)
//   cargo run -p day31 --features solution -- implicit  (the model solution)

use imgcourse::*;
use crate::implicit::*;

/// `f(x, y)` sampled on a `size × size` grid covering `[−half, half]²` (y down).
fn sample(size: usize, half: f32, f: impl Fn(f32, f32) -> f32) -> Image<f32> {
    let h = 2.0 * half / (size - 1) as f32;
    Image::from_fn(size, size, |i, j| f(-half + i as f32 * h, -half + j as f32 * h))
}

/// The number of separate pieces of the region `φ < c` (4-neighbours, flood fill).
fn pieces(phi: &Image<f32>, c: f32) -> usize {
    let mut seen = Image::new(phi.width(), phi.height(), false);
    let mut count = 0;
    for (x, y) in phi.coords() {
        if phi[(x, y)] >= c || seen[(x, y)] {
            continue;
        }
        count += 1;
        let mut stack = vec![(x, y)];
        seen[(x, y)] = true;
        while let Some((x, y)) = stack.pop() {
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if phi.contains(nx, ny) && phi[(nx as usize, ny as usize)] < c && !seen[(nx as usize, ny as usize)] {
                    seen[(nx as usize, ny as usize)] = true;
                    stack.push((nx as usize, ny as usize));
                }
            }
        }
    }
    count
}

/// Blue inside (φ < c), red outside, darker towards the level `c`, black on the level line.
fn picture(phi: &Image<f32>, c: f32) -> Image<[f32; 3]> {
    let m = phi.pixels().iter().fold(0.0f32, |m, v| m.max((v - c).abs()));
    let mut pic = phi.map(|v| {
        let s = ((v - c).abs() / m).sqrt();
        if v < c { [0.3 + 0.2 * s, 0.5 + 0.3 * s, 1.0] } else { [1.0, 0.5 + 0.3 * s, 0.3 + 0.2 * s] }
    });
    for y in 0..phi.height() {
        for x in row_crossings(&phi.map(|v| v - c), y) {
            pic[(x.round() as usize, y)] = [0.0; 3];
        }
    }
    let t = Image::from_fn(phi.height(), phi.width(), |x, y| phi[(y, x)] - c); // columns as rows
    for x in 0..t.height() {
        for y in row_crossings(&t, x) {
            pic[(x, y.round() as usize)] = [0.0; 3];
        }
    }
    pic
}

pub fn main() {
    let dir = "out/implicit";

    // Two circles of radius 1 at (±1.2, 0): the union, cut at different heights c.
    let (size, half) = (241, 3.0);
    let circle = |cx: f32, r: f32| sample(size, half, move |x, y| ((x - cx).powi(2) + y * y).sqrt() - r);
    let two = union(&circle(-1.2, 1.0), &circle(1.2, 1.0));
    for c in [0.0, 0.1, 0.19, 0.21, 0.3] {
        println!("two circles, level c = {c:.2}: {} piece(s)", pieces(&two, c));
    }
    save_rgb(&picture(&two, 0.1), format!("{dir}/two-circles-c0.1.png"));
    save_rgb(&picture(&two, 0.3), format!("{dir}/two-circles-c0.3.png"));
    // A circle with a bite out of it.
    save_rgb(&picture(&difference(&circle(-0.4, 1.4), &circle(0.9, 0.8)), 0.0), format!("{dir}/difference.png"));

    // The curvature of the level line through (1, 0) of φ = √(x² + y²) − 1.5, a circle of
    // radius 1 (exact κ = 1), on grids of decreasing spacing h: the coarsest is the 5 × 5 table.
    for (n, h) in [(5, 1.0), (9, 0.5), (17, 0.25), (41, 0.1)] {
        let half = h * (n - 1) as f32 / 2.0;
        let phi = sample(n, half, |x, y| (x * x + y * y).sqrt() - 1.5);
        let kappa = curvature(&phi);
        let i = ((1.0 + half) / h).round() as usize; // the grid point (1, 0)
        println!("level line through (1, 0), spacing h = {h}: kappa = {:.3} (exact 1)", kappa[(i, n / 2)] / h);
    }

    // A three-lobed shape r = 1 + 0.2 cos 3θ: curvature at a tip and in a dent.
    let (size, half) = (201, 1.5);
    let h = 2.0 * half / (size - 1) as f32;
    let lobes = sample(size, half, |x, y| (x * x + y * y).sqrt() - (1.0 + 0.2 * (3.0 * y.atan2(x)).cos()));
    let kappa = curvature(&lobes).map(|k| k / h);
    let at = |x: f32, y: f32| kappa[(((x + half) / h).round() as usize, ((y + half) / h).round() as usize)];
    let dent = (0.8 * (std::f32::consts::PI / 3.0).cos(), 0.8 * (std::f32::consts::PI / 3.0).sin());
    println!("three lobes: kappa at the tip (1.2, 0) = {:.2} (exact 2.08), in the dent ({:.2}, {:.2}) = {:.2} (exact -1.56)", at(1.2, 0.0), dent.0, dent.1, at(dent.0, dent.1));
    // κ along the curve, as a picture: the curve coloured by its curvature.
    let mut pic = gray_to_rgb(&lobes.map(|v| if v < 0.0 { 0.85 } else { 1.0 }));
    for (x, y) in lobes.coords() {
        if lobes[(x, y)].abs() < 1.2 * h {
            let k = (kappa[(x, y)] / 2.1).clamp(-1.0, 1.0);
            pic[(x, y)] = if k > 0.0 { [k, 0.0, 0.0] } else { [0.0, 0.0, -k] };
        }
    }
    save_rgb(&pic, format!("{dir}/lobes-curvature.png"));
    println!("wrote the pictures to {dir}/");
}
