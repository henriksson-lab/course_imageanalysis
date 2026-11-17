// Model solution for the coding exercises of "Aligning point sets: iterative closest point".

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
pub fn centroid(points: &[Point]) -> Point {
    let (mut sx, mut sy) = (0.0, 0.0);
    for &(x, y) in points {
        sx += x;
        sy += y;
    }
    let n = points.len() as f32;
    (sx / n, sy / n)
}

/// C2. The best rigid transform for known pairs `p[i] ↔ q[i]` (least squares, closed form):
/// centre both sets; the angle is `atan2(S_cross, S_dot)` with `S_dot = Σ p·q` and
/// `S_cross = Σ (p_x q_y − p_y q_x)` over the centred points; the shift is `q̄ − R p̄`.
pub fn fit_rigid(p: &[Point], q: &[Point]) -> Rigid {
    let (px, py) = centroid(p);
    let (qx, qy) = centroid(q);
    let (mut s_dot, mut s_cross) = (0.0, 0.0);
    for (&(x, y), &(u, v)) in p.iter().zip(q) {
        let (x, y, u, v) = (x - px, y - py, u - qx, v - qy);
        s_dot += x * u + y * v;
        s_cross += x * v - y * u;
    }
    let angle = s_cross.atan2(s_dot);
    let turned = Rigid { angle, tx: 0.0, ty: 0.0 }.apply((px, py));
    Rigid { angle, tx: qx - turned.0, ty: qy - turned.1 }
}

/// C3. The nearest point of `set` to `p` (brute force): its index and its distance.
pub fn nearest(p: Point, set: &[Point]) -> (usize, f32) {
    let mut best = (0, f32::INFINITY);
    for (j, &(x, y)) in set.iter().enumerate() {
        let d = (x - p.0).hypot(y - p.1);
        if d < best.1 {
            best = (j, d);
        }
    }
    best
}

/// C4. One ICP iteration from `current`: move every moving point by `current`, match it to its
/// nearest fixed point, keep only the closest `keep` fraction of the pairs (round(keep · n),
/// at least 2; keep = 1 keeps all), and fit a rigid transform from the ORIGINAL moving points
/// to their matched fixed points.
pub fn icp_step(moving: &[Point], fixed: &[Point], current: &Rigid, keep: f32) -> Step {
    let found: Vec<(usize, f32)> = moving.iter().map(|&m| nearest(current.apply(m), fixed)).collect();
    // The closest pairs first (ties: the lower index first).
    let mut order: Vec<usize> = (0..moving.len()).collect();
    order.sort_by(|&a, &b| found[a].1.total_cmp(&found[b].1));
    let n_keep = ((keep * moving.len() as f32).round() as usize).clamp(2.min(moving.len()), moving.len());
    let mut matches = vec![None; moving.len()];
    for &i in &order[..n_keep] {
        matches[i] = Some(found[i].0);
    }
    let (mut p, mut q, mut sum) = (Vec::new(), Vec::new(), 0.0);
    for (i, m) in matches.iter().enumerate() {
        if let Some(j) = *m {
            p.push(moving[i]);
            q.push(fixed[j]);
            sum += found[i].1 * found[i].1;
        }
    }
    Step { transform: fit_rigid(&p, &q), rms: (sum / n_keep as f32).sqrt(), matches }
}

/// C5. ICP: repeat `icp_step` from `start` until the matching no longer changes (then the
/// transform cannot change either), at most `max_iter` times. Returns the final transform and
/// the RMS of every iteration.
pub fn icp(moving: &[Point], fixed: &[Point], start: Rigid, max_iter: usize, keep: f32) -> (Rigid, Vec<f32>) {
    let mut t = start;
    let mut rms = Vec::new();
    let mut previous: Option<Vec<Option<usize>>> = None;
    for _ in 0..max_iter {
        let step = icp_step(moving, fixed, &t, keep);
        rms.push(step.rms);
        t = step.transform;
        if previous.as_ref() == Some(&step.matches) {
            break;
        }
        previous = Some(step.matches);
    }
    (t, rms)
}
