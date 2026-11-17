// Model solution for the coding exercises of "Hypothesise and verify: RANSAC".
// Points are (x, y) in pixels, y down. A line is in normal form x cos θ + y sin θ = ρ with
// θ in [0, π) (radians) and ρ signed, as in the Hough line unit.

use imgcourse::linalg::least_squares;
use imgcourse::Rng;
use std::f32::consts::PI;

/// A line in normal form.
#[derive(Clone, Copy, Debug)]
pub struct Line {
    pub theta: f32,
    pub rho: f32,
}

impl Line {
    /// The line with unit normal `n` through the point `p`, with θ brought into [0, π).
    pub fn from_normal(n: [f32; 2], p: [f32; 2]) -> Line {
        let mut theta = n[1].atan2(n[0]);
        let mut sign = 1.0;
        if theta < 0.0 {
            theta += PI;
            sign = -1.0;
        }
        if theta >= PI {
            theta -= PI;
            sign = -sign;
        }
        Line { theta, rho: sign * (n[0] * p[0] + n[1] * p[1]) }
    }
}

/// C1a. The line through two points: the normal is perpendicular to `q − p`, scaled to length 1.
pub fn line_through(p: [f32; 2], q: [f32; 2]) -> Line {
    let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
    let len = dx.hypot(dy);
    Line::from_normal([-dy / len, dx / len], p)
}

/// C1b. The distance of point `p` from the line: `|x cos θ + y sin θ − ρ|`.
pub fn distance(line: &Line, p: [f32; 2]) -> f32 {
    (p[0] * line.theta.cos() + p[1] * line.theta.sin() - line.rho).abs()
}

/// C2. The number of samples needed for probability `p` of at least one all-inlier sample, with
/// inlier fraction `w` and sample size `s`: `log(1 − p) / log(1 − w^s)`, rounded up.
pub fn iterations_needed(w: f32, s: u32, p: f32) -> usize {
    ((1.0 - p).ln() / (1.0 - w.powi(s as i32)).ln()).ceil() as usize
}

/// C3. The total-least-squares line through points: through their centroid, along the direction
/// of largest spread. With the scatter sums `sxx`, `sxy`, `syy` of the centred points, that
/// direction has the angle `α = ½ atan2(2 sxy, sxx − syy)`; the normal is perpendicular to it.
pub fn fit_line_tls(points: &[[f32; 2]]) -> Line {
    let n = points.len() as f32;
    let (mx, my) = (points.iter().map(|p| p[0]).sum::<f32>() / n, points.iter().map(|p| p[1]).sum::<f32>() / n);
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for p in points {
        let (dx, dy) = (p[0] - mx, p[1] - my);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    let alpha = 0.5 * (2.0 * sxy).atan2(sxx - syy); // direction of the line
    Line::from_normal([-alpha.sin(), alpha.cos()], [mx, my])
}

/// C4. RANSAC for a line: `iterations` times, pick two different points at random, take the line
/// through them, and count the points within `eps` of it; keep the line with the most inliers.
/// Finally refit with `fit_line_tls` on its inliers. Returns the refitted line and the indices of
/// the inliers of the best sample.
pub fn ransac_line(points: &[[f32; 2]], eps: f32, iterations: usize, rng: &mut Rng) -> (Line, Vec<usize>) {
    let mut best: Vec<usize> = Vec::new();
    for _ in 0..iterations {
        let i = rng.below(points.len());
        let mut j = rng.below(points.len());
        while j == i {
            j = rng.below(points.len());
        }
        let line = line_through(points[i], points[j]);
        let inliers: Vec<usize> = (0..points.len()).filter(|&k| distance(&line, points[k]) <= eps).collect();
        if inliers.len() > best.len() {
            best = inliers;
        }
    }
    let chosen: Vec<[f32; 2]> = best.iter().map(|&k| points[k]).collect();
    (fit_line_tls(&chosen), best)
}

/// C5. The algebraic circle fit: solve `x² + y² = 2a x + 2b y + c` for `(a, b, c)` by least
/// squares (one row `(2x, 2y, 1)` per point), then `r = sqrt(c + a² + b²)`. Returns
/// `(a, b, r)`, or `None` if the points do not determine a circle (fewer than 3, or on a line).
pub fn fit_circle(points: &[[f32; 2]]) -> Option<(f32, f32, f32)> {
    let rows: Vec<Vec<f64>> = points.iter().map(|p| vec![2.0 * p[0] as f64, 2.0 * p[1] as f64, 1.0]).collect();
    let rhs: Vec<f64> = points.iter().map(|p| (p[0] as f64).powi(2) + (p[1] as f64).powi(2)).collect();
    if points.len() < 3 {
        return None;
    }
    let sol = least_squares(&rows, &rhs)?;
    let (a, b, c) = (sol[0], sol[1], sol[2]);
    let r2 = c + a * a + b * b;
    if r2 <= 0.0 {
        return None;
    }
    Some((a as f32, b as f32, r2.sqrt() as f32))
}
