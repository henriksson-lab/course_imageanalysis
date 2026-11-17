// Model solution for the coding exercises of "Many views: structure from motion and bundle adjustment".
// Flatland: points (x, y) in a plane, 1D cameras with a position and a heading.

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
    let mut z = z0;
    let mut out = vec![z];
    for _ in 0..steps {
        let r = x_obs - big_x / z;
        let slope = big_x / (z * z);
        z -= r / slope;
        out.push(z);
    }
    out
}

/// C2a. The pixel at which the camera sees the point: in camera coordinates the depth is
/// (p − c)·(cos θ, sin θ) and the sideways offset (p − c)·(sin θ, −cos θ); then
/// u = FOCAL · offset / depth + CENTRE_U.
pub fn project(cam: &Cam, p: &Pt) -> f64 {
    let (dx, dy) = (p[0] - cam[0], p[1] - cam[1]);
    let (s, c) = cam[2].sin_cos();
    let depth = dx * c + dy * s;
    let side = dx * s - dy * c;
    FOCAL * side / depth + CENTRE_U
}

/// C2b. The residual of every observation: observed pixel minus predicted pixel.
pub fn residuals(cams: &[Cam], points: &[Pt], obs: &[Obs]) -> Vec<f64> {
    obs.iter().map(|o| o.u - project(&cams[o.cam], &points[o.point])).collect()
}

/// C3. The Huber loss with threshold δ: returns `(cost, weight)`. The cost is r² for |r| ≤ δ
/// and 2δ|r| − δ² beyond; the weight (for reweighted least squares) is 1 for |r| ≤ δ and δ/|r|
/// beyond, so that the pull 2 · weight · r is capped at 2δ.
pub fn huber(r: f64, delta: f64) -> (f64, f64) {
    if r.abs() <= delta {
        (r * r, 1.0)
    } else {
        (2.0 * delta * r.abs() - delta * delta, delta / r.abs())
    }
}

/// C4. One damped Gauss–Newton step (slide "The math: Gauss–Newton and Levenberg–Marquardt"):
/// solve (JᵀWJ + μ D) δ = −JᵀW r, with W the diagonal of the weights `w` and D the diagonal of
/// JᵀWJ. With μ = 0 this is the Gauss–Newton step. `None` if the system is singular.
pub fn damped_step(j: &[Vec<f64>], r: &[f64], w: &[f64], mu: f64) -> Option<Vec<f64>> {
    let n = j[0].len();
    let mut a = vec![vec![0.0; n]; n];
    let mut b = vec![0.0; n];
    for k in 0..r.len() {
        for p in 0..n {
            b[p] -= j[k][p] * w[k] * r[k];
            for q in 0..n {
                a[p][q] += j[k][p] * w[k] * j[k][q];
            }
        }
    }
    for p in 0..n {
        a[p][p] *= 1.0 + mu;
    }
    solve(&a, &b)
}

/// C5. Bundle adjustment: `iterations` times, compute the residuals, their weights (all 1, or
/// from `huber` if `delta` is given) and the Jacobian, take a `damped_step`, and add it to the
/// free parameters. Gauss–Newton (`levenberg = false`) uses μ = 0 and accepts every step.
/// Levenberg–Marquardt starts with μ = 0.001; a step that lowers the (Huber) cost is accepted
/// and μ divided by 10, otherwise it is rejected and μ multiplied by 10. Changes `cams` and
/// `points`; returns the RMS of the residuals at the start and after each iteration.
pub fn bundle_adjust(cams: &mut [Cam], points: &mut [Pt], obs: &[Obs], iterations: usize, levenberg: bool, delta: Option<f64>) -> Vec<f64> {
    let cost = |r: &[f64]| -> f64 { r.iter().map(|&v| delta.map_or(v * v, |d| huber(v, d).0)).sum() };
    let mut mu = if levenberg { 1e-3 } else { 0.0 };
    let mut r = residuals(cams, points, obs);
    let mut history = vec![rms(&r)];
    for _ in 0..iterations {
        let w: Vec<f64> = r.iter().map(|&v| delta.map_or(1.0, |d| huber(v, d).1)).collect();
        let j = jacobian(cams, points, obs);
        let mut theta = pack(cams, points);
        if let Some(step) = damped_step(&j, &r, &w, mu) {
            for (n, s) in step.iter().enumerate() {
                theta[FIXED + n] += s;
            }
            let (mut c, mut p) = (cams.to_vec(), points.to_vec());
            unpack(&theta, &mut c, &mut p);
            let r_new = residuals(&c, &p, obs);
            if !levenberg || cost(&r_new) < cost(&r) {
                cams.copy_from_slice(&c);
                points.copy_from_slice(&p);
                r = r_new;
                mu /= 10.0;
            } else {
                mu *= 10.0;
            }
        } else if levenberg {
            mu *= 10.0; // a singular system: damp more
        }
        history.push(rms(&r));
    }
    history
}
