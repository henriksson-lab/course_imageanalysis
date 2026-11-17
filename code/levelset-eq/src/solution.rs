// Model solution for the coding exercises of "Moving a curve: the level set equation".

use imgcourse::{Border, Image};

/// C1. The upwind length of the gradient at pixel `(x, y)` (slide "The math: the upwind
/// (Godunov) scheme"), for a front that moves outward (`outward = true`, `F > 0`) or inward.
/// With `D⁻ = φ(x) − φ(x−1)` and `D⁺ = φ(x+1) − φ(x)` (and the same in y):
/// outward: `max(D⁻, 0)² + min(D⁺, 0)²`, inward: `min(D⁻, 0)² + max(D⁺, 0)²`, summed over x
/// and y, then the square root. Outside the image: copy the edge.
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

/// C2. One forward-Euler step of the level set equation `φ_t + F |∇φ| = 0` (slide "One time
/// step: forward Euler"), with the speed `F` given at every pixel: `φ − Δt F |∇φ|`, where
/// `|∇φ|` is the upwind gradient for the sign of `F` at that pixel. Reads only the old φ.
pub fn level_set_step(phi: &Image<f32>, speed: &Image<f32>, dt: f32) -> Image<f32> {
    phi.map_xy(|x, y, v| {
        let f = speed[(x, y)];
        v - dt * f * upwind_gradient(phi, x, y, f > 0.0)
    })
}

/// C3. One step of curvature flow, `F = −εκ` (slide "Speeds: grass fire and curvature flow"):
/// `φ + Δt ε κ |∇φ|`, with central differences, where
/// `κ |∇φ| = (φ_xx φ_y² − 2 φ_x φ_y φ_xy + φ_yy φ_x²) / (φ_x² + φ_y²)` (0 where the gradient is 0).
pub fn curvature_step(phi: &Image<f32>, eps: f32, dt: f32) -> Image<f32> {
    phi.map_xy(|x, y, c| {
        let (x, y) = (x as isize, y as isize);
        let at = |dx: isize, dy: isize| phi.get_border(x + dx, y + dy, Border::Clamp);
        let px = (at(1, 0) - at(-1, 0)) / 2.0;
        let py = (at(0, 1) - at(0, -1)) / 2.0;
        let pxx = at(1, 0) - 2.0 * c + at(-1, 0);
        let pyy = at(0, 1) - 2.0 * c + at(0, -1);
        let pxy = (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1)) / 4.0;
        let g2 = px * px + py * py;
        if g2 < 1e-12 {
            return c;
        }
        c + dt * eps * (pxx * py * py - 2.0 * px * py * pxy + pyy * px * px) / g2
    })
}

/// C4. Reinitialisation by recomputing distances (slide "Keeping φ healthy", method (a)):
/// find every point where the zero level crosses an edge between two neighbouring pixels
/// (linear interpolation, as in the previous unit), then give every pixel the distance to the
/// nearest such point, with the sign of φ. Slow (every pixel against every point), and a
/// little too large between the crossing points (up to about 0.2 pixels), but the zero level
/// stays where it was.
pub fn reinitialise(phi: &Image<f32>) -> Image<f32> {
    let mut points = Vec::new();
    for (x, y) in phi.coords() {
        let a = phi[(x, y)];
        for (dx, dy) in [(1, 0), (0, 1)] {
            if x + dx < phi.width() && y + dy < phi.height() {
                let b = phi[(x + dx, y + dy)];
                if (a < 0.0) != (b < 0.0) {
                    let t = a / (a - b);
                    points.push([x as f32 + t * dx as f32, y as f32 + t * dy as f32]);
                }
            }
        }
    }
    phi.map_xy(|x, y, v| {
        let mut best = f32::INFINITY;
        for p in &points {
            best = best.min((p[0] - x as f32).powi(2) + (p[1] - y as f32).powi(2));
        }
        if v < 0.0 { -best.sqrt() } else { best.sqrt() }
    })
}
