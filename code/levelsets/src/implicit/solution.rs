// Model solution for the coding exercises of "Curves as level sets".

use imgcourse::{Border, Image};

/// C1a. Union of two shapes (slide "Set operations for free"): inside either, `min(φ1, φ2)`.
pub fn union(a: &Image<f32>, b: &Image<f32>) -> Image<f32> {
    a.zip_map(b, |p, q| p.min(q))
}

/// C1b. Intersection: inside both, `max(φ1, φ2)`.
pub fn intersection(a: &Image<f32>, b: &Image<f32>) -> Image<f32> {
    a.zip_map(b, |p, q| p.max(q))
}

/// C1c. Difference `A \ B`: inside A and outside B, `max(φ1, −φ2)`.
pub fn difference(a: &Image<f32>, b: &Image<f32>) -> Image<f32> {
    a.zip_map(b, |p, q| p.max(-q))
}

/// C2. Where the curve crosses row `y`: for every pair of neighbours `x`, `x + 1` whose
/// values have opposite signs, the crossing by linear interpolation, `x + φ(x) / (φ(x) − φ(x+1))`
/// (in pixels, from left to right). A value of exactly 0 counts as a crossing at that pixel.
pub fn row_crossings(phi: &Image<f32>, y: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for x in 0..phi.width() - 1 {
        let (a, b) = (phi[(x, y)], phi[(x + 1, y)]);
        if a == 0.0 {
            out.push(x as f32);
        } else if a * b < 0.0 {
            out.push(x as f32 + a / (a - b));
        }
    }
    if phi[(phi.width() - 1, y)] == 0.0 {
        out.push((phi.width() - 1) as f32);
    }
    out
}

/// C3. The outward unit normal `∇φ / |∇φ|` at pixel `(x, y)` from central differences
/// `φ_x = (φ(x+1, y) − φ(x−1, y)) / 2` (likewise `φ_y`), and `|∇φ|`. Neighbours outside the
/// image: copy the edge.
pub fn normal(phi: &Image<f32>, x: usize, y: usize) -> ([f32; 2], f32) {
    let (x, y) = (x as isize, y as isize);
    let at = |dx: isize, dy: isize| phi.get_border(x + dx, y + dy, Border::Clamp);
    let px = (at(1, 0) - at(-1, 0)) / 2.0;
    let py = (at(0, 1) - at(0, -1)) / 2.0;
    let len = (px * px + py * py).sqrt();
    if len == 0.0 {
        return ([0.0, 0.0], 0.0);
    }
    ([px / len, py / len], len)
}

/// C4. The curvature of the level line through every pixel (slide "The math: normal and
/// curvature"):
/// `κ = (φ_xx φ_y² − 2 φ_x φ_y φ_xy + φ_yy φ_x²) / (φ_x² + φ_y²)^(3/2)`, all derivatives by
/// central differences, `φ_xy = (φ(x+1, y+1) − φ(x+1, y−1) − φ(x−1, y+1) + φ(x−1, y−1)) / 4`.
/// Where the gradient is 0, `κ = 0`.
pub fn curvature(phi: &Image<f32>) -> Image<f32> {
    phi.map_xy(|x, y, c| {
        let (x, y) = (x as isize, y as isize);
        let at = |dx: isize, dy: isize| phi.get_border(x + dx, y + dy, Border::Clamp);
        let px = (at(1, 0) - at(-1, 0)) / 2.0;
        let py = (at(0, 1) - at(0, -1)) / 2.0;
        let pxx = at(1, 0) - 2.0 * c + at(-1, 0);
        let pyy = at(0, 1) - 2.0 * c + at(0, -1);
        let pxy = (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1)) / 4.0;
        let g2 = px * px + py * py;
        if g2 == 0.0 {
            return 0.0;
        }
        (pxx * py * py - 2.0 * px * py * pxy + pyy * px * px) / g2.powf(1.5)
    })
}
