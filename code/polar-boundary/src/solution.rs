// Model solution for the coding exercises of "Closed boundaries: DP in polar coordinates".

use imgcourse::{Border, Image};
use std::f32::consts::PI;

/// Which edge the cost looks for, going outward along a ray.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Polarity {
    /// Bright inside, dark outside (a bright nucleus on a dark background).
    BrightToDark,
    /// Dark inside, bright outside.
    DarkToBright,
    /// Either kind of edge.
    Either,
}

/// C1. Bilinear interpolation: the grey value at a position `(x, y)` between pixel centres,
/// from the four pixels around it. Outside the image, the edge pixels are copied.
pub fn bilinear(img: &Image<f32>, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (x0, y0) = (x0 as isize, y0 as isize);
    let p = |dx: isize, dy: isize| img.get_border(x0 + dx, y0 + dy, Border::Clamp);
    let top = (1.0 - fx) * p(0, 0) + fx * p(1, 0);
    let bottom = (1.0 - fx) * p(0, 1) + fx * p(1, 1);
    (1.0 - fy) * top + fy * bottom
}

/// C2. Unwrap the image around the centre `(cx, cy)`: the polar image has one column per
/// angle `θ_k = 2πk / angles` (k = 0 .. angles − 1) and one row per radius r = 0 .. radii − 1,
/// with `P(k, r) = I(cx + r cos θ_k, cy + r sin θ_k)`. Because y points down, the angle turns
/// clockwise on the screen.
pub fn unwrap(img: &Image<f32>, cx: f32, cy: f32, angles: usize, radii: usize) -> Image<f32> {
    Image::from_fn(angles, radii, |k, r| {
        let theta = 2.0 * PI * k as f32 / angles as f32;
        let r = r as f32;
        bilinear(img, cx + r * theta.cos(), cy + r * theta.sin())
    })
}

/// C3. The edge cost along every ray (every column of the polar image): the radial
/// derivative `∂P/∂r` by central differences (one-sided at the first and last radius),
/// with the sign that makes the wanted edge cheap (negative).
pub fn radial_cost(polar: &Image<f32>, polarity: Polarity) -> Image<f32> {
    let last = polar.height() - 1;
    polar.map_xy(|k, r, _| {
        let (r0, r1) = (r.saturating_sub(1), (r + 1).min(last));
        let derivative = (polar[(k, r1)] - polar[(k, r0)]) / (r1 - r0) as f32;
        match polarity {
            Polarity::BrightToDark => derivative,
            Polarity::DarkToBright => -derivative,
            Polarity::Either => -derivative.abs(),
        }
    })
}

/// C4. The DP table over the angles: column k is filled from column k − 1 with
/// `M(k, r) = C(k, r) + min over |d| ≤ delta of ( M(k − 1, r + d) + alpha·|d| )`.
/// Returns `M` and the pointers `d`. With `start = Some(s)`, the path must start at radius
/// `s`: `M(0, r) = C(0, r)` for r = s and ∞ for every other r.
pub fn polar_table(cost: &Image<f32>, delta: usize, alpha: f32, start: Option<usize>) -> (Image<f32>, Image<isize>) {
    let (k_count, r_count) = (cost.width(), cost.height());
    let mut m = Image::new(k_count, r_count, f32::INFINITY);
    let mut pointer = Image::new(k_count, r_count, 0isize);
    for r in 0..r_count {
        if start.is_none() || start == Some(r) {
            m[(0, r)] = cost[(0, r)];
        }
    }
    let delta = delta as isize;
    for k in 1..k_count {
        for r in 0..r_count {
            let mut best = f32::INFINITY;
            let mut best_d = 0;
            for d in -delta..=delta {
                let ra = r as isize + d;
                if ra < 0 || ra >= r_count as isize {
                    continue;
                }
                let candidate = m[(k - 1, ra as usize)] + alpha * d.abs() as f32;
                if candidate < best {
                    best = candidate;
                    best_d = d;
                }
            }
            m[(k, r)] = cost[(k, r)] + best;
            pointer[(k, r)] = best_d;
        }
    }
    (m, pointer)
}

/// Follow the pointers back from radius `end` in the last column: the radius at every angle.
fn follow(pointer: &Image<isize>, end: usize) -> Vec<usize> {
    let k_count = pointer.width();
    let mut radii = vec![0; k_count];
    let mut r = end;
    for k in (0..k_count).rev() {
        radii[k] = r;
        if k > 0 {
            r = (r as isize + pointer[(k, r)]) as usize;
        }
    }
    radii
}

/// C5a. The best open path: the smallest entry of the last column of the table (no start
/// fixed), backtracked. Returns the radius index at every angle and the path's cost.
pub fn open_path(cost: &Image<f32>, delta: usize, alpha: f32) -> (Vec<usize>, f32) {
    let (m, pointer) = polar_table(cost, delta, alpha, None);
    let last = cost.width() - 1;
    let mut end = 0;
    for r in 0..cost.height() {
        if m[(last, r)] < m[(last, end)] {
            end = r;
        }
    }
    (follow(&pointer, end), m[(last, end)])
}

/// C5b. The best closed path (exact): for every start radius s, fill the table with only s
/// allowed at angle 0, and close it with an end radius r within `delta` of s, paying
/// `alpha·|r − s|` for the closing step. Keep the cheapest. Returns the radius index at every
/// angle and the cost, closing step included.
pub fn closed_path(cost: &Image<f32>, delta: usize, alpha: f32) -> (Vec<usize>, f32) {
    let last = cost.width() - 1;
    let mut best = (Vec::new(), f32::INFINITY);
    for s in 0..cost.height() {
        let (m, pointer) = polar_table(cost, delta, alpha, Some(s));
        for r in s.saturating_sub(delta)..=(s + delta).min(cost.height() - 1) {
            let total = m[(last, r)] + alpha * r.abs_diff(s) as f32;
            if total < best.1 {
                best = (follow(&pointer, r), total);
            }
        }
    }
    best
}
