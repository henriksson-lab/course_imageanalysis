// Coding exercises for "Shape models: landmarks, Procrustes and PCA".
// Replace each `todo!()` with your code, then check it with `cargo test -p day30 shape_models::` and look at
// `cargo run -p day30 -- shape-models`.
#![allow(unused_variables, unused_imports)]

use imgcourse::linalg;

/// A landmark `[x, y]`.
pub type Point = [f64; 2];

/// A similarity transform `p ↦ M p + t` with `M = [[a, −b], [b, a]]`: scale `√(a² + b²)`,
/// angle `atan2(b, a)`, then the shift `t`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Similarity {
    pub a: f64,
    pub b: f64,
    pub t: Point,
}

/// C1. Procrustes (slides "Procrustes: three steps" and "The math: Procrustes in closed
/// form"): the similarity that maps the landmarks `p` best onto `q` (least squares).
/// Centre both sets, then `a = Σ p·q / Σ |p|²`, `b = Σ (p_x q_y − p_y q_x) / Σ |p|²` on the
/// centred points, and `t = centroid(q) − M centroid(p)`. With `allow_scale = false` the fit is
/// rigid: `(a, b)` is scaled to length 1.
///
/// Hint: `centroid` (below) gives the centroids. One loop over the landmark pairs adds up
/// `Σ |p|²`, `Σ p·q` and `Σ (p_x q_y − p_y q_x)` of the centred points.
pub fn procrustes(p: &[Point], q: &[Point], allow_scale: bool) -> Similarity {
    todo!("C1: procrustes")
}

/// C2. Generalised Procrustes analysis (slide "Aligning a whole training set"). Returns the
/// mean shape and all shapes aligned to it. Centre every shape and scale it to size 1; take
/// the first as the mean; then `rounds` times: align every shape to the mean, average them
/// landmark by landmark, and make the new mean centred, of size 1 and aligned to the first
/// guess (rigidly, so that it cannot turn or drift from round to round).
///
/// Hint: `normalise`, `procrustes` (C1) and `apply` do the work; the rounds only average.
pub fn gpa(shapes: &[Vec<Point>], rounds: usize) -> (Vec<Point>, Vec<Vec<Point>>) {
    todo!("C2: gpa")
}

/// C3. The point distribution model (slide "The math: the point distribution model"): the
/// mean vector, and the eigenvalues (largest first) and unit eigenvectors (the modes) of the
/// covariance matrix `S = (1/s) Σ (x − mean)(x − mean)ᵀ` of the `s` shape vectors.
///
/// Hint: build the `d × d` covariance as `Vec<Vec<f64>>`, then `linalg::sym_eigen(&cov)` gives
/// the eigenvalues (largest first) and the eigenvectors.
pub fn shape_model(vectors: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    todo!("C3: shape_model")
}

/// C4a. A model shape: `mean + Σ_j b[j] · modes[j]` (uses as many modes as `b` has numbers).
pub fn generate(mean: &[f64], modes: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    todo!("C4a: generate")
}

/// C4b. The parameters of the model shape closest to `x`, using the first `k` modes:
/// `b[j] = modes[j] · (x − mean)`. With `clamp`, each `b[j]` is limited to `±3 √values[j]`.
///
/// Hint: `b[j]` is a dot product of `modes[j]` with `x − mean`.
pub fn project(x: &[f64], mean: &[f64], modes: &[Vec<f64>], values: &[f64], k: usize, clamp: bool) -> Vec<f64> {
    todo!("C4b: project")
}

// ---- Helpers (filled in, not part of the exercise) ----

/// The centroid (mean landmark).
pub fn centroid(p: &[Point]) -> Point {
    let n = p.len() as f64;
    [p.iter().map(|q| q[0]).sum::<f64>() / n, p.iter().map(|q| q[1]).sum::<f64>() / n]
}

/// Apply a similarity to every landmark.
pub fn apply(s: Similarity, p: &[Point]) -> Vec<Point> {
    p.iter().map(|q| [s.a * q[0] - s.b * q[1] + s.t[0], s.b * q[0] + s.a * q[1] + s.t[1]]).collect()
}

/// Centred at the origin and scaled to size 1 (sum of squared distances to the centroid = 1).
pub fn normalise(p: &[Point]) -> Vec<Point> {
    let c = centroid(p);
    let size = p.iter().map(|q| (q[0] - c[0]).powi(2) + (q[1] - c[1]).powi(2)).sum::<f64>().sqrt();
    p.iter().map(|q| [(q[0] - c[0]) / size, (q[1] - c[1]) / size]).collect()
}

/// The sum of squared distances between corresponding landmarks.
pub fn sum_sq(p: &[Point], q: &[Point]) -> f64 {
    p.iter().zip(q).map(|(a, b)| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sum()
}

/// Landmarks as one vector `(x1, y1, x2, y2, …)`, and back.
pub fn flatten(p: &[Point]) -> Vec<f64> {
    p.iter().flat_map(|q| [q[0], q[1]]).collect()
}

pub fn unflatten(v: &[f64]) -> Vec<Point> {
    v.chunks(2).map(|c| [c[0], c[1]]).collect()
}
