//! Random point sets and drawing for the RANSAC exercises. Not part of the exercise.

use imgcourse::{Image, Rng};

/// `n_in` points on the line x cos θ + y sin θ = ρ (θ in degrees) inside the square
/// [0, size)², moved across the line by Gaussian noise of sd `noise`, then `n_out` outliers
/// spread uniformly over the square. The inliers come first.
pub fn line_points(size: f32, theta_deg: f32, rho: f32, noise: f32, n_in: usize, n_out: usize, rng: &mut Rng) -> Vec<[f32; 2]> {
    let t = theta_deg.to_radians();
    let (n, d) = ([t.cos(), t.sin()], [-t.sin(), t.cos()]);
    let mut pts = Vec::new();
    while pts.len() < n_in {
        let s = (rng.uniform() - 0.5) * 2.0 * size;
        let e = noise * rng.normal();
        let p = [rho * n[0] + s * d[0] + e * n[0], rho * n[1] + s * d[1] + e * n[1]];
        if (0.0..size).contains(&p[0]) && (0.0..size).contains(&p[1]) {
            pts.push(p);
        }
    }
    for _ in 0..n_out {
        pts.push([size * rng.uniform(), size * rng.uniform()]);
    }
    pts
}

/// `n_in` points on the circle with centre `[a, b]` and radius r, moved along the radius by
/// noise of sd `noise`, then `n_out` uniform outliers in [0, size)².
pub fn circle_points(size: f32, [a, b]: [f32; 2], r: f32, noise: f32, n_in: usize, n_out: usize, rng: &mut Rng) -> Vec<[f32; 2]> {
    let mut pts = Vec::new();
    for _ in 0..n_in {
        let t = std::f32::consts::TAU * rng.uniform();
        let rr = r + noise * rng.normal();
        pts.push([a + rr * t.cos(), b + rr * t.sin()]);
    }
    for _ in 0..n_out {
        pts.push([size * rng.uniform(), size * rng.uniform()]);
    }
    pts
}

/// A white picture of a `size × size` square, enlarged `scale` times.
pub fn canvas(size: f32, scale: f32) -> Image<[f32; 3]> {
    let n = (size * scale) as usize;
    Image::new(n, n, [1.0, 1.0, 1.0])
}

/// A small filled square at point `p` (in square coordinates) in colour `c`.
pub fn dot(img: &mut Image<[f32; 3]>, p: [f32; 2], scale: f32, c: [f32; 3]) {
    let (x, y) = ((p[0] * scale).round() as isize, (p[1] * scale).round() as isize);
    for dy in -2..=2 {
        for dx in -2..=2 {
            if img.contains(x + dx, y + dy) {
                img[((x + dx) as usize, (y + dy) as usize)] = c;
            }
        }
    }
}

/// Draw the points of the curve `f(t)`, t in [0, 1], in colour `c` (dashed if `dashed`).
pub fn curve(img: &mut Image<[f32; 3]>, scale: f32, c: [f32; 3], dashed: bool, f: impl Fn(f32) -> [f32; 2]) {
    let n = 4000;
    for i in 0..n {
        if dashed && (i / 40) % 2 == 1 {
            continue;
        }
        let p = f(i as f32 / n as f32);
        let (x, y) = ((p[0] * scale).round() as isize, (p[1] * scale).round() as isize);
        if img.contains(x, y) {
            img[(x as usize, y as usize)] = c;
        }
    }
}
