// Model solution for the coding exercises of "Active shape models".

use imgcourse::Image;

/// A landmark `[x, y]`, in pixels.
pub type Point = [f32; 2];

/// A similarity transform `p ↦ M p + t` with `M = [[a, −b], [b, a]]` (the pose).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub a: f32,
    pub b: f32,
    pub t: Point,
}

/// A point distribution model (previous unit): `x = mean + Σ_j b[j] modes[j]`, with the
/// shape vectors written `(x1, y1, x2, y2, …)` in the model frame.
pub struct ShapeModel {
    pub mean: Vec<f32>,
    pub modes: Vec<Vec<f32>>,
    pub values: Vec<f32>,
}

/// C1a. The strongest-edge rule (slide "Looking along the normal"): the offset, from `−m` to
/// `m`, of the largest value of `profile` (which has `2m + 1` values, offsets `−m..=m`).
pub fn strongest_edge(profile: &[f32]) -> isize {
    let m = (profile.len() / 2) as isize;
    let mut best = 0;
    for i in 0..profile.len() {
        if profile[i] > profile[best] {
            best = i;
        }
    }
    best as isize - m
}

/// C1b. Profile matching (slide "A profile by hand"): slide the model (`2k + 1` values) over
/// the profile (`2m + 1` values, offsets `−m..=m`) and add the squared differences, for every
/// centre from `−(m − k)` to `m − k`. Returns the best centre and all the costs, in order.
pub fn match_profile(profile: &[f32], model: &[f32]) -> (isize, Vec<f32>) {
    let (m, k) = ((profile.len() / 2) as isize, (model.len() / 2) as isize);
    let mut costs = Vec::new();
    for centre in -(m - k)..=(m - k) {
        let mut cost = 0.0;
        for j in -k..=k {
            let d = profile[(m + centre + j) as usize] - model[(k + j) as usize];
            cost += d * d;
        }
        costs.push(cost);
    }
    let mut best = 0;
    for i in 0..costs.len() {
        if costs[i] < costs[best] {
            best = i;
        }
    }
    (best as isize - (m - k), costs)
}

/// C2. Project and clamp (slide "The search loop", steps 4–5): `b[j] = modes[j] · (y − mean)`,
/// each limited to `±3 √values[j]`. Returns `b` and the model shape `mean + Σ b[j] modes[j]`.
pub fn project_clamp(y: &[f32], model: &ShapeModel) -> (Vec<f32>, Vec<f32>) {
    let mut b = vec![0.0; model.modes.len()];
    let mut shape = model.mean.clone();
    for (j, phi) in model.modes.iter().enumerate() {
        for i in 0..y.len() {
            b[j] += phi[i] * (y[i] - model.mean[i]);
        }
        let limit = 3.0 * model.values[j].sqrt();
        b[j] = b[j].clamp(-limit, limit);
        for i in 0..y.len() {
            shape[i] += b[j] * phi[i];
        }
    }
    (b, shape)
}

/// C3. The outward unit normal at every landmark of a closed outline: perpendicular to the
/// line from the previous to the next landmark, turned to point away from the centroid.
pub fn normals(points: &[Point]) -> Vec<Point> {
    let n = points.len();
    let c = centroid(points);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let (prev, next) = (points[(i + n - 1) % n], points[(i + 1) % n]);
        let (tx, ty) = (next[0] - prev[0], next[1] - prev[1]);
        let len = (tx * tx + ty * ty).sqrt();
        let mut normal = [ty / len, -tx / len];
        if normal[0] * (points[i][0] - c[0]) + normal[1] * (points[i][1] - c[1]) < 0.0 {
            normal = [-normal[0], -normal[1]];
        }
        out.push(normal);
    }
    out
}

/// C4. Suggest a better position for every landmark (step 2 of the search loop): sample the
/// derivative of the image along the normal at offsets `−(m + k)..=(m + k)` (`k` = half the
/// model length), `g[o] = I(p + (o + ½) n) − I(p + (o − ½) n)`, divide by `Σ |g|`, find the
/// best centre with `match_profile`, and move the landmark that far along its normal.
pub fn suggest(img: &Image<f32>, points: &[Point], m: usize, model_profile: &[f32]) -> Vec<Point> {
    let k = model_profile.len() / 2;
    let r = (m + k) as isize;
    let mut out = Vec::with_capacity(points.len());
    for (p, n) in points.iter().zip(normals(points)) {
        let at = |s: f32| bilinear(img, p[0] + s * n[0], p[1] + s * n[1]);
        let mut g: Vec<f32> = (-r..=r).map(|o| at(o as f32 + 0.5) - at(o as f32 - 0.5)).collect();
        let total: f32 = g.iter().map(|v| v.abs()).sum::<f32>().max(1e-12);
        g.iter_mut().for_each(|v| *v /= total);
        let (best, _) = match_profile(&g, model_profile);
        out.push([p[0] + best as f32 * n[0], p[1] + best as f32 * n[1]]);
    }
    out
}

/// C5. One iteration of the search (slide "The search loop"): the model shape in the image
/// `X = T(mean + Φ b)`, suggestions `Y` (C4), the pose that maps the model shape onto `Y`
/// (Procrustes), `y = T⁻¹(Y)` in the model frame, then project and clamp (C2). Returns the new
/// pose and `b`.
pub fn asm_iteration(img: &Image<f32>, model: &ShapeModel, pose: Pose, b: &[f32], m: usize, model_profile: &[f32]) -> (Pose, Vec<f32>) {
    let shape = to_points(&model_shape(model, b));
    let x = apply(pose, &shape);
    let y = suggest(img, &x, m, model_profile);
    let pose = procrustes(&shape, &y);
    let y_model = apply(inverse(pose), &y);
    let (b, _) = project_clamp(&to_vector(&y_model), model);
    (pose, b)
}

// ---- Helpers from earlier units (filled in, not part of the exercise) ----

/// The model shape `mean + Σ b[j] modes[j]` (without clamping).
pub fn model_shape(model: &ShapeModel, b: &[f32]) -> Vec<f32> {
    let mut x = model.mean.clone();
    for (bj, phi) in b.iter().zip(&model.modes) {
        for i in 0..x.len() {
            x[i] += bj * phi[i];
        }
    }
    x
}

/// The pose applied to every landmark.
pub fn apply(s: Pose, p: &[Point]) -> Vec<Point> {
    p.iter().map(|q| [s.a * q[0] - s.b * q[1] + s.t[0], s.b * q[0] + s.a * q[1] + s.t[1]]).collect()
}

/// The inverse pose: `M⁻¹ = [[a, b], [−b, a]] / (a² + b²)`, `t' = −M⁻¹ t`.
pub fn inverse(s: Pose) -> Pose {
    let d = s.a * s.a + s.b * s.b;
    let (a, b) = (s.a / d, -s.b / d);
    Pose { a, b, t: [-(a * s.t[0] - b * s.t[1]), -(b * s.t[0] + a * s.t[1])] }
}

/// Procrustes (previous unit): the similarity that maps `p` best onto `q`.
pub fn procrustes(p: &[Point], q: &[Point]) -> Pose {
    let (cp, cq) = (centroid(p), centroid(q));
    let (mut pp, mut dot, mut cross) = (0.0, 0.0, 0.0);
    for (pi, qi) in p.iter().zip(q) {
        let (px, py, qx, qy) = (pi[0] - cp[0], pi[1] - cp[1], qi[0] - cq[0], qi[1] - cq[1]);
        pp += px * px + py * py;
        dot += px * qx + py * qy;
        cross += px * qy - py * qx;
    }
    let (a, b) = (dot / pp, cross / pp);
    Pose { a, b, t: [cq[0] - (a * cp[0] - b * cp[1]), cq[1] - (b * cp[0] + a * cp[1])] }
}

pub fn centroid(p: &[Point]) -> Point {
    let n = p.len() as f32;
    [p.iter().map(|q| q[0]).sum::<f32>() / n, p.iter().map(|q| q[1]).sum::<f32>() / n]
}

pub fn to_points(v: &[f32]) -> Vec<Point> {
    v.chunks(2).map(|c| [c[0], c[1]]).collect()
}

pub fn to_vector(p: &[Point]) -> Vec<f32> {
    p.iter().flat_map(|q| [q[0], q[1]]).collect()
}

/// The image value between pixels (bilinear); outside, the nearest edge pixel. In `src/common.rs`.
pub use crate::common::bilinear;
