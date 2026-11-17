// Coding exercises for "Active shape models". Replace each `todo!()` with your code,
// then check it with `cargo test -p snakes asm::` and look at `cargo run -p snakes -- asm`.
#![allow(unused_variables, unused_imports)]

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
    todo!("C1a: strongest_edge")
}

/// C1b. Profile matching (slide "A profile by hand"): slide the model (`2k + 1` values) over
/// the profile (`2m + 1` values, offsets `−m..=m`) and add the squared differences, for every
/// centre from `−(m − k)` to `m − k`. Returns the best centre and all the costs, in order.
///
/// Hint: with `m = profile.len() / 2` and `k = model.len() / 2`, offset `o` of the profile is
/// `profile[(m + o) as usize]`; work with `isize` offsets.
pub fn match_profile(profile: &[f32], model: &[f32]) -> (isize, Vec<f32>) {
    todo!("C1b: match_profile")
}

/// C2. Project and clamp (slide "The search loop", steps 4–5): `b[j] = modes[j] · (y − mean)`,
/// each limited to `±3 √values[j]`. Returns `b` and the model shape `mean + Σ b[j] modes[j]`.
///
/// Hint: the same as `project` and `generate` of the previous unit, with the clamp.
pub fn project_clamp(y: &[f32], model: &ShapeModel) -> (Vec<f32>, Vec<f32>) {
    todo!("C2: project_clamp")
}

/// C3. The outward unit normal at every landmark of a closed outline: perpendicular to the
/// line from the previous to the next landmark, turned to point away from the centroid.
///
/// Hint: the tangent is `next − prev`; turning `(tx, ty)` by 90° gives `(ty, −tx)`. Check the
/// sign with the dot product of the normal and `point − centroid`.
pub fn normals(points: &[Point]) -> Vec<Point> {
    todo!("C3: normals")
}

/// C4. Suggest a better position for every landmark (step 2 of the search loop): sample the
/// derivative of the image along the normal at offsets `−(m + k)..=(m + k)` (`k` = half the
/// model length), `g[o] = I(p + (o + ½) n) − I(p + (o − ½) n)`, divide by `Σ |g|`, find the
/// best centre with `match_profile`, and move the landmark that far along its normal.
///
/// Hint: `bilinear(img, x, y)` reads the image between pixels; `match_profile` (C1b) gives the
/// offset.
pub fn suggest(img: &Image<f32>, points: &[Point], m: usize, model_profile: &[f32]) -> Vec<Point> {
    todo!("C4: suggest")
}

/// C5. One iteration of the search (slide "The search loop"): the model shape in the image
/// `X = T(mean + Φ b)`, suggestions `Y` (C4), the pose that maps the model shape onto `Y`
/// (Procrustes), `y = T⁻¹(Y)` in the model frame, then project and clamp (C2). Returns the new
/// pose and `b`.
///
/// Hint: `model_shape`, `to_points`, `apply`, `procrustes`, `inverse` and `to_vector` are below.
pub fn asm_iteration(img: &Image<f32>, model: &ShapeModel, pose: Pose, b: &[f32], m: usize, model_profile: &[f32]) -> (Pose, Vec<f32>) {
    todo!("C5: asm_iteration")
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
