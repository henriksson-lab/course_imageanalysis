// Coding exercises for "Meshes, coordinate spaces and projection".
// Replace each `todo!()` with your code, then check it with `cargo test -p graphics3d pipeline3d::`
// and look at `cargo run -p graphics3d -- pipeline3d`.
// Conventions of the lecture (OpenGL): the camera looks down −z, y up; NDC in [−1, 1]³;
// screen pixels with the origin at the bottom left.
#![allow(unused_variables, unused_imports)]

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
///
/// Hint: `sub` and `cross` are given above.
pub fn face_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    todo!("C1: face_normal")
}

/// C2a. The matrix product `a b` (apply `b` first, then `a`).
///
/// Hint: entry (i, j) of the product is the sum over k of `a[i][k] * b[k][j]`.
pub fn mat_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    todo!("C2a: mat_mul")
}

/// C2b. The matrix `m` applied to the homogeneous vector `v`.
pub fn apply(m: &Mat4, v: Vec4) -> Vec4 {
    todo!("C2b: apply")
}

/// C3. The view matrix of a camera at `eye` looking at `target`, with `up` roughly upwards
/// (slide "The math: the look-at matrix"): the rows of R are r, u′ and −f, and the last
/// column is −R e.
///
/// Hint: f = normalize(target − eye), r = normalize(f × up), u′ = r × f. Start from
/// `IDENTITY`, put r, u′ and −f in the first three rows, and `-dot(row, eye)` in the last column.
pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    todo!("C3: look_at")
}

/// C4. The perspective matrix for a vertical field of view `fov_deg` (degrees), the aspect
/// ratio width/height, and the near and far distances (slide "The math: the perspective
/// matrix"). It puts −z into w, and maps z = −near to depth −1 and z = −far to +1.
///
/// Hint: s = 1 / tan(fov / 2) with the angle in radians (`to_radians()`);
/// A = (far + near) / (far − near), B = 2 far near / (far − near); the third row is (0, 0, −A, −B).
pub fn perspective(fov_deg: f64, aspect: f64, near: f64, far: f64) -> Mat4 {
    todo!("C4: perspective")
}

/// C5. From clip coordinates to the screen: divide by w (NDC), then map x and y from
/// [−1, 1] to [0, width] and [0, height] pixels (origin at the bottom left). Returns
/// `(x, y, depth)` with the NDC depth, or `None` for a point behind the camera (w ≤ 0).
///
/// Hint: check `clip[3] <= 0.0` first.
pub fn to_screen(clip: Vec4, width: f64, height: f64) -> Option<Vec3> {
    todo!("C5: to_screen")
}
