// Model solution for the coding exercises of "Edge-based segmentation: geodesic active contours".

use imgcourse::{Border, Image};

/// C1. The edge-stopping function (slide "The image as a speed") of an image that is already
/// smoothed: `g = 1 / (1 + |∇I|² / k²)`, with central differences `(I(x+1) − I(x−1)) / 2` in x
/// and y (outside the image: copy the edge).
pub fn edge_stopping(smoothed: &Image<f32>, k: f32) -> Image<f32> {
    smoothed.map_xy(|x, y, _| {
        let (x, y) = (x as isize, y as isize);
        let at = |dx: isize, dy: isize| smoothed.get_border(x + dx, y + dy, Border::Clamp);
        let gx = (at(1, 0) - at(-1, 0)) / 2.0;
        let gy = (at(0, 1) - at(0, -1)) / 2.0;
        1.0 / (1.0 + (gx * gx + gy * gy) / (k * k))
    })
}

/// C2. One step of the braking balloon (slide "Stage 1: a balloon that brakes"), `φ < 0` inside:
/// `φ − Δt g c |∇φ|  +  Δt ε g κ|∇φ|`. The balloon term uses the upwind gradient for the sign of
/// `c` (`upwind_gradient`, given); the curvature term central differences, with
/// `κ|∇φ| = (φ_xx φ_y² − 2 φ_x φ_y φ_xy + φ_yy φ_x²) / (φ_x² + φ_y²)` (0 where the gradient is 0).
pub fn braking_step(phi: &Image<f32>, g: &Image<f32>, c: f32, eps: f32, dt: f32) -> Image<f32> {
    phi.map_xy(|x, y, v| {
        let balloon = g[(x, y)] * c * upwind_gradient(phi, x, y, c > 0.0);
        let (x, y) = (x as isize, y as isize);
        let at = |dx: isize, dy: isize| phi.get_border(x + dx, y + dy, Border::Clamp);
        let px = (at(1, 0) - at(-1, 0)) / 2.0;
        let py = (at(0, 1) - at(0, -1)) / 2.0;
        let pxx = at(1, 0) - 2.0 * v + at(-1, 0);
        let pyy = at(0, 1) - 2.0 * v + at(0, -1);
        let pxy = (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1)) / 4.0;
        let g2 = px * px + py * py;
        let curv = if g2 < 1e-12 { 0.0 } else { (pxx * py * py - 2.0 * px * py * pxy + pyy * px * px) / g2 };
        v - dt * balloon + dt * eps * g[(x as usize, y as usize)] * curv
    })
}

/// C3. One step of the pull into the edge (slide "The math: the geodesic active contour flow"):
/// transport of φ with the velocity `v = −a ∇g` (central differences for `∇g`):
/// `φ − Δt (v_x φ_x + v_y φ_y)`, where `φ_x` is the **upwind** difference: the backward one
/// `φ(x) − φ(x−1)` if `v_x > 0` (the values come from the left), else the forward one
/// `φ(x+1) − φ(x)`; the same for y.
pub fn advection_step(phi: &Image<f32>, g: &Image<f32>, a: f32, dt: f32) -> Image<f32> {
    phi.map_xy(|x, y, v| {
        let (x, y) = (x as isize, y as isize);
        let gat = |dx: isize, dy: isize| g.get_border(x + dx, y + dy, Border::Clamp);
        let at = |dx: isize, dy: isize| phi.get_border(x + dx, y + dy, Border::Clamp);
        let vx = -a * (gat(1, 0) - gat(-1, 0)) / 2.0;
        let vy = -a * (gat(0, 1) - gat(0, -1)) / 2.0;
        let px = if vx > 0.0 { v - at(-1, 0) } else { at(1, 0) - v };
        let py = if vy > 0.0 { v - at(0, -1) } else { at(0, 1) - v };
        v - dt * (vx * px + vy * py)
    })
}

// ---- Helpers from the previous units (filled in, not part of the exercise) ----

/// The upwind `|∇φ|` for a front moving outward (`true`) or inward (unit "The level set
/// equation").
pub fn upwind_gradient(phi: &Image<f32>, x: usize, y: usize, outward: bool) -> f32 {
    let (xi, yi) = (x as isize, y as isize);
    let at = |dx: isize, dy: isize| phi.get_border(xi + dx, yi + dy, Border::Clamp);
    let c = at(0, 0);
    let mut sum = 0.0;
    for (back, forward) in [(c - at(-1, 0), at(1, 0) - c), (c - at(0, -1), at(0, 1) - c)] {
        if outward {
            sum += back.max(0.0).powi(2) + forward.min(0.0).powi(2);
        } else {
            sum += back.min(0.0).powi(2) + forward.max(0.0).powi(2);
        }
    }
    sum.sqrt()
}

/// Gaussian blur of width `sigma` in two passes, copying the edge (`sigma = 0`: a copy).
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
