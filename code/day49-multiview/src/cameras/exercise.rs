// Coding exercises for "The pinhole camera and calibration".
// Replace each `todo!()` with your code, then check it with `cargo test -p day49 cameras::`
// and look at `cargo run -p day49 -- cameras`.
// Vision conventions: the camera looks down +z, x right, y down; pixels (u, v) with u to the
// right and v down. All numbers in f64.
#![allow(unused_variables, unused_imports)]

use imgcourse::linalg::sym_eigen;

/// 3-vectors and 3 × 3 matrices (row by row: `m[row][column]`) with `mat_vec`, `mat_mul`,
/// `transpose` and `dot`, in `src/common.rs`.
pub use crate::common::{dot, mat_mul, mat_vec, transpose, Mat3, Vec3};

/// A 3 × 4 camera matrix P = K [R | t], row by row.
pub type Mat34 = [[f64; 4]; 3];

/// A camera with square pixels and no skew: focal length `f` in pixels, principal point
/// `(cx, cy)`, and the extrinsics: camera coordinates = `r · X + t`.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub f: f64,
    pub cx: f64,
    pub cy: f64,
    pub r: Mat3,
    pub t: Vec3,
}

/// The rotation and translation of a camera at `eye` looking at `target`, with `up` roughly
/// upwards in the world: the rows of R are the camera's right, down and forward axes, and
/// t = −R · eye.
pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> (Mat3, Vec3) {
    let unit = |v: Vec3| {
        let n = dot(v, v).sqrt();
        v.map(|x| x / n)
    };
    let f = unit([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
    let uf = dot(up, f);
    let down = unit([0, 1, 2].map(|k| uf * f[k] - up[k]));
    let right = [down[1] * f[2] - down[2] * f[1], down[2] * f[0] - down[0] * f[2], down[0] * f[1] - down[1] * f[0]];
    let r = [right, down, f];
    let t = mat_vec(&r, eye).map(|v| -v);
    (r, t)
}

/// The matrix P = K [R | t] of a camera.
pub fn camera_matrix(cam: &Camera) -> Mat34 {
    let k = [[cam.f, 0.0, cam.cx], [0.0, cam.f, cam.cy], [0.0, 0.0, 1.0]];
    let kr = mat_mul(&k, &cam.r);
    let kt = mat_vec(&k, cam.t);
    [0, 1, 2].map(|i| [kr[i][0], kr[i][1], kr[i][2], kt[i]])
}

/// The pixel of the world point `x` under the camera matrix `p` (divide by the third entry).
pub fn project_p(p: &Mat34, x: Vec3) -> [f64; 2] {
    let h = [0, 1, 2].map(|i| p[i][0] * x[0] + p[i][1] * x[1] + p[i][2] * x[2] + p[i][3]);
    [h[0] / h[2], h[1] / h[2]]
}

/// The similarity transform (as a matrix in homogeneous coordinates, and its inverse) that
/// moves `n`-dimensional points to mean 0 and an RMS distance `sqrt(n)` from the origin
/// (Hartley's normalisation). Returns `(T, T⁻¹)`, each `(n + 1) × (n + 1)`.
pub fn normalising_transform(points: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = points[0].len();
    let m = points.len() as f64;
    let mean: Vec<f64> = (0..n).map(|k| points.iter().map(|p| p[k]).sum::<f64>() / m).collect();
    let rms = (points.iter().map(|p| (0..n).map(|k| (p[k] - mean[k]).powi(2)).sum::<f64>()).sum::<f64>() / m).sqrt();
    let s = (n as f64).sqrt() / rms;
    let mut t = vec![vec![0.0; n + 1]; n + 1];
    let mut t_inv = vec![vec![0.0; n + 1]; n + 1];
    for k in 0..n {
        t[k][k] = s;
        t[k][n] = -s * mean[k];
        t_inv[k][k] = 1.0 / s;
        t_inv[k][n] = mean[k];
    }
    t[n][n] = 1.0;
    t_inv[n][n] = 1.0;
    (t, t_inv)
}

/// C1. The pixel `(u, v)` where the camera sees the world point `x`: camera coordinates
/// `(X, Y, Z) = R x + t`, then `u = f X/Z + cx`, `v = f Y/Z + cy`. `None` if the point is not in
/// front of the camera (Z ≤ 0).
///
/// Hint: `mat_vec(&cam.r, x)` gives R x; add t, test Z, then divide.
pub fn project(cam: &Camera, x: Vec3) -> Option<[f64; 2]> {
    todo!("C1: project")
}

/// C2. The ray of the pixel `(u, v)` in world coordinates: the camera centre `c = −Rᵀ t` and
/// the direction `Rᵀ ((u − cx)/f, (v − cy)/f, 1)` (slide "From a pixel back to a ray").
/// Every point `c + s · direction`, s > 0, lands on this pixel.
///
/// Hint: `transpose(&cam.r)` is Rᵀ, the inverse of a rotation.
pub fn pixel_ray(cam: &Camera, u: f64, v: f64) -> (Vec3, Vec3) {
    todo!("C2: pixel_ray")
}

/// C3a. Radial distortion of normalised camera coordinates (x, y) = (X/Z, Y/Z):
/// multiply by `1 + k1 r² + k2 r⁴` with `r² = x² + y²`.
pub fn distort(x: f64, y: f64, k1: f64, k2: f64) -> [f64; 2] {
    todo!("C3a: distort")
}

/// C3b. Undo the radial distortion by fixed-point steps: start with (x, y) = (xd, yd) and
/// repeat (x, y) ← (xd, yd) / (1 + k1 r² + k2 r⁴), with r² from the current (x, y), 20 times.
///
/// Hint: r² is computed from the current estimate (x, y), not from (xd, yd).
pub fn undistort(xd: f64, yd: f64, k1: f64, k2: f64) -> [f64; 2] {
    todo!("C3b: undistort")
}

/// C4. The direct linear transform: the camera matrix P (up to scale) from n ≥ 6 world
/// points and their pixels (slide "The math: the DLT"). Normalise the points first, build two
/// rows of A per point, take the unit vector p that makes |A p| smallest (the eigenvector of
/// AᵀA with the smallest eigenvalue), and undo the normalisation: P = T⁻¹ P̂ U.
///
/// Hint: `normalising_transform` gives T for the pixels (3 × 3) and U for the world points
/// (4 × 4). With the normalised point X̂ (4 entries) and pixel (u, v), the two rows of A are
/// (0ᵀ, −X̂ᵀ, v X̂ᵀ) and (X̂ᵀ, 0ᵀ, −u X̂ᵀ). Form AᵀA (12 × 12); `sym_eigen` returns the eigenvectors
/// sorted with the largest eigenvalue first, so the one you need is the last. Its entries 0–3,
/// 4–7 and 8–11 are the rows of P̂.
pub fn dlt(points: &[Vec3], pixels: &[[f64; 2]]) -> Mat34 {
    todo!("C4: dlt")
}

/// C5. Split a camera matrix into K (upper triangular, positive diagonal, K[2][2] = 1), a
/// rotation R and t, with P ∝ K [R | t]: Gram–Schmidt on the rows of the left 3 × 3 block
/// M = K R, from the last row upwards (an RQ decomposition). If det M < 0, flip the sign of P
/// first. Then t = K⁻¹ (last column), with P scaled so that K[2][2] = 1.
///
/// Hint: work from row 2 up to row 0: subtract from row i of M its components along the rows
/// of R already found (those dot products are K[i][j]); the length of what is left is K[i][i],
/// and the rest divided by it is row i of R.
pub fn decompose(p: &Mat34) -> (Mat3, Mat3, Vec3) {
    todo!("C5: decompose")
}
