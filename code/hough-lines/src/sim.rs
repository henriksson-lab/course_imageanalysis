//! Test scenes, edge points and drawing for the Hough exercises.
//! Not part of the exercise: read it if you like.

use imgcourse::{Border, Image, Rng};

/// An edge point: position in pixels, gradient direction `theta` in radians
/// (`atan2(gy, gx)`, pointing from dark to bright), and gradient strength.
#[derive(Clone, Copy, Debug)]
pub struct EdgePoint {
    pub x: f32,
    pub y: f32,
    pub theta: f32,
    pub strength: f32,
}

/// Is `(x, y)` inside the polygon (corners in order)? Even–odd rule.
pub fn inside(poly: &[[f32; 2]], x: f32, y: f32) -> bool {
    let mut c = false;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        if (a[1] > y) != (b[1] > y) && x < a[0] + (y - a[1]) / (b[1] - a[1]) * (b[0] - a[0]) {
            c = !c;
        }
    }
    c
}

/// Fill polygons with the given grey values on a background, then add Gaussian noise.
pub fn polygon_scene(size: usize, background: f32, shapes: &[(Vec<[f32; 2]>, f32)], noise: f32, seed: u64) -> Image<f32> {
    let mut rng = Rng::new(seed);
    Image::from_fn(size, size, |x, y| {
        let (px, py) = (x as f32, y as f32);
        let mut v = background;
        for (poly, grey) in shapes {
            if inside(poly, px, py) {
                v = *grey;
            }
        }
        v + noise * rng.normal()
    })
}

/// Edge points, a simplified Canny (the edges day): blur with a Gaussian of width `sigma`, Sobel
/// gradient, keep pixels whose gradient magnitude is a maximum across the edge (along the
/// gradient, rounded to one of four directions) and above `threshold` times the largest magnitude.
pub fn edge_points(img: &Image<f32>, sigma: f32, threshold: f32) -> Vec<EdgePoint> {
    let r = (3.0 * sigma).ceil() as isize;
    let w: Vec<f32> = (-r..=r).map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = w.iter().sum();
    let pass = |src: &Image<f32>, along_x: bool| {
        src.map_xy(|x, y, _| {
            let mut sum = 0.0;
            for (i, wi) in w.iter().enumerate() {
                let d = i as isize - r;
                let (xx, yy) = if along_x { (x as isize + d, y as isize) } else { (x as isize, y as isize + d) };
                sum += wi / s * src.get_border(xx, yy, Border::Clamp);
            }
            sum
        })
    };
    let b = pass(&pass(img, true), false);
    let at = |x: isize, y: isize| b.get_border(x, y, Border::Clamp);
    let grad = b.map_xy(|x, y, _| {
        let (x, y) = (x as isize, y as isize);
        let gx = at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2.0 * at(x - 1, y) - at(x - 1, y + 1);
        let gy = at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2.0 * at(x, y - 1) - at(x + 1, y - 1);
        [gx, gy]
    });
    let mag = grad.map(|g| g[0].hypot(g[1]));
    let t = threshold * mag.max();
    let mut out = Vec::new();
    for (x, y) in img.coords() {
        let m = mag[(x, y)];
        if m <= t || x == 0 || y == 0 || x + 1 == img.width() || y + 1 == img.height() {
            continue;
        }
        let [gx, gy] = grad[(x, y)];
        let a = gy.atan2(gx);
        // the neighbour along the gradient, rounded to 0°, 45°, 90° or 135°
        let k = ((a / std::f32::consts::FRAC_PI_4).round() as i32).rem_euclid(4);
        let (dx, dy) = [(1, 0), (1, 1), (0, 1), (-1, 1)][k as usize];
        let n1 = mag[((x as isize + dx) as usize, (y as isize + dy) as usize)];
        let n2 = mag[((x as isize - dx) as usize, (y as isize - dy) as usize)];
        if m >= n1 && m > n2 {
            out.push(EdgePoint { x: x as f32, y: y as f32, theta: a, strength: m });
        }
    }
    out
}

/// A grey image as colour, with edge points in orange.
pub fn show_points(img: &Image<f32>, pts: &[EdgePoint]) -> Image<[f32; 3]> {
    let (lo, hi) = (img.min(), img.max());
    let mut out = img.map(|v| {
        let g = 0.6 * (v - lo) / (hi - lo).max(1e-6);
        [g, g, g]
    });
    for p in pts {
        out[(p.x as usize, p.y as usize)] = [1.0, 0.55, 0.0];
    }
    out
}

/// Draw a straight piece from `a` to `b` in colour `c`.
pub fn draw_segment(img: &mut Image<[f32; 3]>, a: [f32; 2], b: [f32; 2], c: [f32; 3]) {
    let n = ((b[0] - a[0]).abs().max((b[1] - a[1]).abs()).ceil() as usize).max(1);
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let (x, y) = ((a[0] + t * (b[0] - a[0])).round() as isize, (a[1] + t * (b[1] - a[1])).round() as isize);
        if img.contains(x, y) {
            img[(x as usize, y as usize)] = c;
        }
    }
}

/// Draw the whole line `x cos θ + y sin θ = ρ` (coordinates relative to `centre`) across the image.
pub fn draw_line(img: &mut Image<[f32; 3]>, theta: f32, rho: f32, centre: [f32; 2], c: [f32; 3]) {
    let (n, d) = ([theta.cos(), theta.sin()], [-theta.sin(), theta.cos()]);
    let p = [centre[0] + rho * n[0], centre[1] + rho * n[1]];
    let l = 2.0 * (img.width() + img.height()) as f32;
    draw_segment(img, [p[0] - l * d[0], p[1] - l * d[1]], [p[0] + l * d[0], p[1] + l * d[1]], c);
}
