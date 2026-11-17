// Model solution for the coding exercises of "Active contours (snakes)".

use imgcourse::{Border, Image};

/// A point `[x, y]` of the snake (a bead), in pixels.
pub type Point = [f32; 2];

/// C1. The internal force on bead `i` of a closed snake (slide "The math: the snake energy
/// and its force"): elastic `α (x[i-1] − 2 x[i] + x[i+1])` plus rigid
/// `−β (x[i-2] − 4 x[i-1] + 6 x[i] − 4 x[i+1] + x[i+2])`. Indices wrap around: after the
/// last bead comes the first.
pub fn internal_force(points: &[Point], i: usize, alpha: f32, beta: f32) -> Point {
    let n = points.len();
    let at = |k: isize| points[(i as isize + k).rem_euclid(n as isize) as usize];
    let mut force = [0.0; 2];
    for c in 0..2 {
        let (m2, m1, x, p1, p2) = (at(-2)[c], at(-1)[c], at(0)[c], at(1)[c], at(2)[c]);
        let elastic = alpha * (m1 - 2.0 * x + p1);
        let rigid = -beta * (m2 - 4.0 * m1 + 6.0 * x - 4.0 * p1 + p2);
        force[c] = elastic + rigid;
    }
    force
}

/// C2. The edge map `f = |∇(G_σ * I)|²` (slide "The image term"): blur, take the gradient
/// with central differences, square its length. Scaled so that its largest value is 1, so
/// that the image weight `w` does not depend on the contrast of the image.
pub fn edge_map(img: &Image<f32>, sigma: f32) -> Image<f32> {
    let smooth = gaussian_blur(img, sigma);
    let mut f = Image::new(img.width(), img.height(), 0.0);
    for (x, y) in img.coords() {
        let (x, y) = (x as isize, y as isize);
        let gx = (smooth.get_border(x + 1, y, Border::Clamp) - smooth.get_border(x - 1, y, Border::Clamp)) / 2.0;
        let gy = (smooth.get_border(x, y + 1, Border::Clamp) - smooth.get_border(x, y - 1, Border::Clamp)) / 2.0;
        f[(x as usize, y as usize)] = gx * gx + gy * gy;
    }
    let max = f.max();
    if max > 0.0 {
        f = f.map(|v| v / max);
    }
    f
}

/// C3. One explicit step of the snake (slide "Moving the snake: one step at a time"):
/// every bead moves by `τ × (internal force + w ∇f)`, with `∇f` read at the bead's
/// position by central differences of bilinear lookups. All forces use the old positions.
pub fn snake_step(points: &[Point], f: &Image<f32>, alpha: f32, beta: f32, w: f32, tau: f32) -> Vec<Point> {
    let mut new = Vec::with_capacity(points.len());
    for (i, p) in points.iter().enumerate() {
        let internal = internal_force(points, i, alpha, beta);
        let (x, y) = (p[0], p[1]);
        let fx = (bilinear(f, x + 1.0, y) - bilinear(f, x - 1.0, y)) / 2.0;
        let fy = (bilinear(f, x, y + 1.0) - bilinear(f, x, y - 1.0)) / 2.0;
        new.push([x + tau * (internal[0] + w * fx), y + tau * (internal[1] + w * fy)]);
    }
    new
}

// ---- Helpers from earlier units (filled in, not part of the exercise) ----

/// Gaussian blur of width `sigma` in two passes (unit "Filtering & convolution"), copying
/// the edge outside the image. `sigma = 0` returns a copy.
pub fn gaussian_blur(img: &Image<f32>, sigma: f32) -> Image<f32> {
    if sigma <= 0.0 {
        return img.clone();
    }
    let r = (3.0 * sigma).ceil() as isize;
    let mut w: Vec<f32> = (-r..=r).map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let total: f32 = w.iter().sum();
    w.iter_mut().for_each(|v| *v /= total);
    let pass = |src: &Image<f32>, along_x: bool| {
        src.map_xy(|x, y, _| {
            let mut sum = 0.0;
            for d in -r..=r {
                let (xx, yy) = if along_x { (x as isize + d, y as isize) } else { (x as isize, y as isize + d) };
                sum += w[(d + r) as usize] * src.get_border(xx, yy, Border::Clamp);
            }
            sum
        })
    };
    pass(&pass(img, true), false)
}

/// The image value at a position between pixels, by bilinear interpolation (unit "Images as
/// arrays"); outside the image the nearest edge pixel is used.
pub fn bilinear(img: &Image<f32>, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (x0, y0) = (x0 as isize, y0 as isize);
    let p = |dx: isize, dy: isize| img.get_border(x0 + dx, y0 + dy, Border::Clamp);
    (1.0 - fx) * (1.0 - fy) * p(0, 0) + fx * (1.0 - fy) * p(1, 0) + (1.0 - fx) * fy * p(0, 1) + fx * fy * p(1, 1)
}

/// `n` beads on a circle of radius `r` around `(cx, cy)`.
pub fn circle(cx: f32, cy: f32, r: f32, n: usize) -> Vec<Point> {
    (0..n)
        .map(|i| {
            let t = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
            [cx + r * t.cos(), cy + r * t.sin()]
        })
        .collect()
}
