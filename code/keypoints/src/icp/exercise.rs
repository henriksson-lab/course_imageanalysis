// Coding exercises for "Aligning point sets: iterative closest point". Replace each `todo!()`
// with your code, then check it with `cargo test -p keypoints icp::` and look at `cargo run -p keypoints -- icp`.
#![allow(unused_variables, unused_imports)]

/// A point `(x, y)`.
pub type Point = (f32, f32);

/// A rigid transform: turn by `angle` (radians; with y down, a positive angle turns clockwise
/// on screen), then shift by `(tx, ty)`: `x' = cos·x − sin·y + tx`, `y' = sin·x + cos·y + ty`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rigid {
    pub angle: f32,
    pub tx: f32,
    pub ty: f32,
}

impl Rigid {
    /// No turn, no shift.
    pub const IDENTITY: Rigid = Rigid { angle: 0.0, tx: 0.0, ty: 0.0 };

    /// Not an exercise: apply the transform to a point.
    pub fn apply(&self, (x, y): Point) -> Point {
        let (s, c) = self.angle.sin_cos();
        (c * x - s * y + self.tx, s * x + c * y + self.ty)
    }
}

/// One ICP iteration: the new transform, the RMS distance of the kept pairs before the fit,
/// and for every moving point the index of its nearest fixed point (`None` if trimming left
/// the pair out).
#[derive(Clone, Debug)]
pub struct Step {
    pub transform: Rigid,
    pub rms: f32,
    pub matches: Vec<Option<usize>>,
}

/// C1. The centroid (mean position) of a set of points.
///
/// Hint: add up the x and the y coordinates, divide by the number of points.
pub fn centroid(points: &[Point]) -> Point {
    todo!("C1: centroid")
}

/// C2. The best rigid transform for known pairs `p[i] ↔ q[i]` (least squares, closed form):
/// centre both sets; the angle is `atan2(S_cross, S_dot)` with `S_dot = Σ p·q` and
/// `S_cross = Σ (p_x q_y − p_y q_x)` over the centred points; the shift is `q̄ − R p̄`.
///
/// Hint: `s_cross.atan2(s_dot)`; for the shift, turn `p̄` with
/// `Rigid { angle, tx: 0.0, ty: 0.0 }.apply(p_bar)` and subtract it from `q̄`.
pub fn fit_rigid(p: &[Point], q: &[Point]) -> Rigid {
    todo!("C2: fit_rigid")
}

/// C3. The nearest point of `set` to `p` (brute force): its index and its distance.
///
/// Hint: start with `(0, f32::INFINITY)` and keep the smaller distance; `(a - b).hypot(c - d)`.
pub fn nearest(p: Point, set: &[Point]) -> (usize, f32) {
    todo!("C3: nearest")
}

/// C4. One ICP iteration from `current`: move every moving point by `current`, match it to its
/// nearest fixed point, keep only the closest `keep` fraction of the pairs (round(keep · n),
/// at least 2; keep = 1 keeps all), and fit a rigid transform from the ORIGINAL moving points
/// to their matched fixed points.
///
/// Hint: collect `nearest(current.apply(m), fixed)` for every moving point; sort the indices
/// `0..n` by distance (`order.sort_by(|&a, &b| d[a].total_cmp(&d[b]))`, a stable sort, so ties
/// keep the lower index first); the first `n_keep` of them are the kept pairs.
pub fn icp_step(moving: &[Point], fixed: &[Point], current: &Rigid, keep: f32) -> Step {
    todo!("C4: icp_step")
}

/// C5. ICP: repeat `icp_step` from `start` until the matching no longer changes (then the
/// transform cannot change either), at most `max_iter` times. Returns the final transform and
/// the RMS of every iteration.
///
/// Hint: remember the previous step's `matches` in an `Option<Vec<Option<usize>>>` and stop
/// (after taking the new transform) when the new matches are equal to it.
pub fn icp(moving: &[Point], fixed: &[Point], start: Rigid, max_iter: usize, keep: f32) -> (Rigid, Vec<f32>) {
    todo!("C5: icp")
}
