// Coding exercises for "Many views: structure from motion and bundle adjustment".
// Replace each `todo!()` with your code, then check it with `cargo test -p day49 sfm::`
// and look at `cargo run -p day49 -- sfm`.
// Flatland: points (x, y) in a plane, 1D cameras with a position and a heading.
#![allow(unused_variables, unused_imports)]

use imgcourse::linalg::solve;

/// Focal length and principal point of the flatland cameras (pixels), as in the demo.
pub const FOCAL: f64 = 130.0;
pub const CENTRE_U: f64 = 80.0;
/// The first 4 parameters (camera 0 entirely, and the x position of camera 1) are held fixed:
/// this removes the 4 gauge freedoms of flatland (shift x, y, rotation, scale).
pub const FIXED: usize = 4;

/// A flatland camera: position (x, y) and heading θ (radians); it looks along (cos θ, sin θ).
pub type Cam = [f64; 3];
/// A flatland point (x, y).
pub type Pt = [f64; 2];

/// One observation: camera `cam` sees point `point` at pixel `u`.
#[derive(Clone, Copy, Debug)]
pub struct Obs {
    pub cam: usize,
    pub point: usize,
    pub u: f64,
}

/// All parameters in one vector: the cameras (3 numbers each), then the points (2 each).
pub fn pack(cams: &[Cam], points: &[Pt]) -> Vec<f64> {
    cams.iter().flat_map(|c| c.iter().copied()).chain(points.iter().flat_map(|p| p.iter().copied())).collect()
}

/// The inverse of `pack`.
pub fn unpack(theta: &[f64], cams: &mut [Cam], points: &mut [Pt]) {
    let m = cams.len();
    for (j, c) in cams.iter_mut().enumerate() {
        *c = [theta[3 * j], theta[3 * j + 1], theta[3 * j + 2]];
    }
    for (i, p) in points.iter_mut().enumerate() {
        *p = [theta[3 * m + 2 * i], theta[3 * m + 2 * i + 1]];
    }
}

/// The Jacobian of the residuals with respect to the free parameters (all but the first
/// `FIXED`), by central differences: `j[k][n]` = ∂ residual k / ∂ parameter FIXED + n.
pub fn jacobian(cams: &[Cam], points: &[Pt], obs: &[Obs]) -> Vec<Vec<f64>> {
    let theta = pack(cams, points);
    let (mut c, mut p) = (cams.to_vec(), points.to_vec());
    let free = theta.len() - FIXED;
    let mut j = vec![vec![0.0; free]; obs.len()];
    let h = 1e-6;
    for n in 0..free {
        let mut t = theta.clone();
        t[FIXED + n] += h;
        unpack(&t, &mut c, &mut p);
        let plus = residuals(&c, &p, obs);
        t[FIXED + n] -= 2.0 * h;
        unpack(&t, &mut c, &mut p);
        let minus = residuals(&c, &p, obs);
        for k in 0..obs.len() {
            j[k][n] = (plus[k] - minus[k]) / (2.0 * h);
        }
    }
    j
}

/// The root mean square of the residuals.
pub fn rms(r: &[f64]) -> f64 {
    (r.iter().map(|v| v * v).sum::<f64>() / r.len() as f64).sqrt()
}

/// C1. Gauss–Newton for the depth Z of a point at sideways position `big_x`, seen at the
/// normalised image coordinate `x_obs`: the residual is r(Z) = x_obs − X/Z, its slope
/// r′(Z) = X/Z², and each step is Z ← Z − r/r′. Returns Z after 0, 1, ..., `steps` steps.
pub fn gauss_newton_depth(x_obs: f64, big_x: f64, z0: f64, steps: usize) -> Vec<f64> {
    todo!("C1: gauss_newton_depth")
}

/// C2a. The pixel at which the camera sees the point: in camera coordinates the depth is
/// (p − c)·(cos θ, sin θ) and the sideways offset (p − c)·(sin θ, −cos θ); then
/// u = FOCAL · offset / depth + CENTRE_U.
///
/// Hint: `let (s, c) = cam[2].sin_cos();`
pub fn project(cam: &Cam, p: &Pt) -> f64 {
    todo!("C2a: project")
}

/// C2b. The residual of every observation: observed pixel minus predicted pixel.
///
/// Hint: `obs.iter().map(|o| ...)` with `cams[o.cam]` and `points[o.point]`.
pub fn residuals(cams: &[Cam], points: &[Pt], obs: &[Obs]) -> Vec<f64> {
    todo!("C2b: residuals")
}

/// C3. The Huber loss with threshold δ: returns `(cost, weight)`. The cost is r² for |r| ≤ δ
/// and 2δ|r| − δ² beyond; the weight (for reweighted least squares) is 1 for |r| ≤ δ and δ/|r|
/// beyond, so that the pull 2 · weight · r is capped at 2δ.
pub fn huber(r: f64, delta: f64) -> (f64, f64) {
    todo!("C3: huber")
}

/// C4. One damped Gauss–Newton step (slide "The math: Gauss–Newton and Levenberg–Marquardt"):
/// solve (JᵀWJ + μ D) δ = −JᵀW r, with W the diagonal of the weights `w` and D the diagonal of
/// JᵀWJ. With μ = 0 this is the Gauss–Newton step. `None` if the system is singular.
///
/// Hint: build A = JᵀWJ (n × n) and b = −JᵀW r with three nested loops, multiply the
/// diagonal of A by (1 + μ), and call `solve(&a, &b)`.
pub fn damped_step(j: &[Vec<f64>], r: &[f64], w: &[f64], mu: f64) -> Option<Vec<f64>> {
    todo!("C4: damped_step")
}

/// C5. Bundle adjustment: `iterations` times, compute the residuals, their weights (all 1, or
/// from `huber` if `delta` is given) and the Jacobian, take a `damped_step`, and add it to the
/// free parameters. Gauss–Newton (`levenberg = false`) uses μ = 0 and accepts every step.
/// Levenberg–Marquardt starts with μ = 0.001; a step that lowers the (Huber) cost is accepted
/// and μ divided by 10, otherwise it is rejected and μ multiplied by 10. Changes `cams` and
/// `points`; returns the RMS of the residuals at the start and after each iteration.
///
/// Hint: `jacobian`, `pack` and `unpack` are given above; the step changes only the
/// parameters from index `FIXED` on. Keep the residuals of the current estimate, and compare the
/// cost of the new residuals with the old (the sum of squares, or of Huber costs).
pub fn bundle_adjust(cams: &mut [Cam], points: &mut [Pt], obs: &[Obs], iterations: usize, levenberg: bool, delta: Option<f64>) -> Vec<f64> {
    todo!("C5: bundle_adjust")
}
