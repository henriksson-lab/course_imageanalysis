// Model solution for the coding exercises of "Meshes, coordinate spaces and projection".
// Conventions of the lecture (OpenGL): the camera looks down −z, y up; NDC in [−1, 1]³;
// screen pixels with the origin at the bottom left.

/// A point or direction in 3D.
pub type Vec3 = [f64; 3];
/// Homogeneous coordinates: `w = 1` for points, `w = 0` for directions.
pub type Vec4 = [f64; 4];
/// A 4 × 4 matrix, row by row: `m[row][column]`.
pub type Mat4 = [[f64; 4]; 4];

pub const IDENTITY: Mat4 = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]];

pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// `dot` and `normalize` of 3D vectors, in `src/common.rs`.
pub use crate::common::{dot, normalize};

/// The translation by `t`.
pub fn translation(t: Vec3) -> Mat4 {
    [[1., 0., 0., t[0]], [0., 1., 0., t[1]], [0., 0., 1., t[2]], [0., 0., 0., 1.]]
}

/// The rotation by `deg` degrees about the z axis (counter-clockwise seen from +z).
pub fn rotation_z(deg: f64) -> Mat4 {
    let (s, c) = deg.to_radians().sin_cos();
    [[c, -s, 0., 0.], [s, c, 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
}

/// The rotation by `deg` degrees about the y axis.
pub fn rotation_y(deg: f64) -> Mat4 {
    let (s, c) = deg.to_radians().sin_cos();
    [[c, 0., s, 0.], [0., 1., 0., 0.], [-s, 0., c, 0.], [0., 0., 0., 1.]]
}

/// C1. The face normal of the triangle `a, b, c`: `(b − a) × (c − a)` (not scaled to length 1).
/// It points out of the front side, the side from which `a, b, c` run counter-clockwise.
pub fn face_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    cross(sub(b, a), sub(c, a))
}

/// C2a. The matrix product `a b` (apply `b` first, then `a`).
pub fn mat_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

/// C2b. The matrix `m` applied to the homogeneous vector `v`.
pub fn apply(m: &Mat4, v: Vec4) -> Vec4 {
    let mut out = [0.0; 4];
    for i in 0..4 {
        for k in 0..4 {
            out[i] += m[i][k] * v[k];
        }
    }
    out
}

/// C3. The view matrix of a camera at `eye` looking at `target`, with `up` roughly upwards
/// (slide "The math: the look-at matrix"): the rows of R are r, u′ and −f, and the last
/// column is −R e.
pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    let f = normalize(sub(target, eye));
    let r = normalize(cross(f, up));
    let u = cross(r, f);
    let rows = [r, u, [-f[0], -f[1], -f[2]]];
    let mut v = IDENTITY;
    for i in 0..3 {
        for j in 0..3 {
            v[i][j] = rows[i][j];
        }
        v[i][3] = -dot(rows[i], eye);
    }
    v
}

/// C4. The perspective matrix for a vertical field of view `fov_deg` (degrees), the aspect
/// ratio width/height, and the near and far distances (slide "The math: the perspective
/// matrix"). It puts −z into w, and maps z = −near to depth −1 and z = −far to +1.
pub fn perspective(fov_deg: f64, aspect: f64, near: f64, far: f64) -> Mat4 {
    let s = 1.0 / (fov_deg.to_radians() / 2.0).tan();
    let a = (far + near) / (far - near);
    let b = 2.0 * far * near / (far - near);
    [[s / aspect, 0., 0., 0.], [0., s, 0., 0.], [0., 0., -a, -b], [0., 0., -1., 0.]]
}

/// C5. From clip coordinates to the screen: divide by w (NDC), then map x and y from
/// [−1, 1] to [0, width] and [0, height] pixels (origin at the bottom left). Returns
/// `(x, y, depth)` with the NDC depth, or `None` for a point behind the camera (w ≤ 0).
pub fn to_screen(clip: Vec4, width: f64, height: f64) -> Option<Vec3> {
    let w = clip[3];
    if w <= 0.0 {
        return None;
    }
    let (x, y, z) = (clip[0] / w, clip[1] / w, clip[2] / w);
    Some([(x + 1.0) / 2.0 * width, (y + 1.0) / 2.0 * height, z])
}
