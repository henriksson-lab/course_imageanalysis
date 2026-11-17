// Coding exercises for "Hypothesise and verify: RANSAC".
// Replace each `todo!()` with your code, then check it with `cargo test -p ransac`
// and look at `cargo run -p ransac`.
#![allow(unused_variables, unused_imports)]

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
///
/// Hint: a normal of the direction `(dx, dy)` is `(−dy, dx)`; divide by the length and
/// use `Line::from_normal`, which brings θ into [0, π) and gives ρ the matching sign.
pub fn line_through(p: [f32; 2], q: [f32; 2]) -> Line {
    todo!("C1a: line_through")
}

/// C1b. The distance of point `p` from the line: `|x cos θ + y sin θ − ρ|`.
pub fn distance(line: &Line, p: [f32; 2]) -> f32 {
    todo!("C1b: distance")
}

/// C2. The number of samples needed for probability `p` of at least one all-inlier sample, with
/// inlier fraction `w` and sample size `s`: `log(1 − p) / log(1 − w^s)`, rounded up.
///
/// Hint: `.ln()`, `.powi(s as i32)` and `.ceil() as usize`.
pub fn iterations_needed(w: f32, s: u32, p: f32) -> usize {
    todo!("C2: iterations_needed")
}

/// C3. The total-least-squares line through points: through their centroid, along the direction
/// of largest spread. With the scatter sums `sxx`, `sxy`, `syy` of the centred points, that
/// direction has the angle `α = ½ atan2(2 sxy, sxx − syy)`; the normal is perpendicular to it.
///
/// Hint: first the centroid, then the three sums; the normal of the direction α is
/// `(−sin α, cos α)`.
pub fn fit_line_tls(points: &[[f32; 2]]) -> Line {
    todo!("C3: fit_line_tls")
}

/// C4. RANSAC for a line: `iterations` times, pick two different points at random, take the line
/// through them, and count the points within `eps` of it; keep the line with the most inliers.
/// Finally refit with `fit_line_tls` on its inliers. Returns the refitted line and the indices of
/// the inliers of the best sample.
///
/// Hint: `rng.below(points.len())` draws an index; draw again while it equals the
/// first. Keep the inlier list of the best sample in a `Vec<usize>`.
pub fn ransac_line(points: &[[f32; 2]], eps: f32, iterations: usize, rng: &mut Rng) -> (Line, Vec<usize>) {
    todo!("C4: ransac_line")
}

/// C5. The algebraic circle fit: solve `x² + y² = 2a x + 2b y + c` for `(a, b, c)` by least
/// squares (one row `(2x, 2y, 1)` per point), then `r = sqrt(c + a² + b²)`. Returns
/// `(a, b, r)`, or `None` if the points do not determine a circle (fewer than 3, or on a line).
///
/// Hint: build the rows `vec![2x, 2y, 1.0]` and the right-hand sides `x² + y²` in
/// `f64`, then `least_squares(&rows, &rhs)?` (it returns `None` for a singular system).
pub fn fit_circle(points: &[[f32; 2]]) -> Option<(f32, f32, f32)> {
    todo!("C5: fit_circle")
}
