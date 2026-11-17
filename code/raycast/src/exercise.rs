// Coding exercises for "Rays from the camera, and where they hit".
// Replace each `todo!()` with your code, then check it with `cargo test -p raycast`
// and look at `cargo run -p raycast`.
#![allow(unused_variables, unused_imports)]

use crate::vec3::{Ray, Vec3};

/// The shapes of the scene.
#[derive(Clone, Copy, Debug)]
pub enum Shape {
    Sphere { centre: Vec3, radius: f32 },
    Plane { point: Vec3, normal: Vec3 },
    Triangle { a: Vec3, b: Vec3, c: Vec3 },
}

/// What a ray hit: the distance `t`, the unit normal (turned towards the eye) and the
/// index of the object in the scene.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub t: f32,
    pub normal: Vec3,
    pub object: usize,
}

/// C1. The camera ray through the centre of pixel `(i, j)` (column, row from the top) of a
/// `width` × `height` image. The camera sits at the origin, looks down −z, `y` is up, and
/// `fov_deg` is the vertical field of view in degrees. The direction has length 1.
///
/// Hint: `xn = 2 (i + 0.5) / width − 1`, `yn = 1 − 2 (j + 0.5) / height`; the direction is `(xn · tan(fov/2) · width/height, yn · tan(fov/2), −1)`,
/// normalised. `f32::to_radians` and `tan` exist; `Ray::new(origin, dir)`.
pub fn camera_ray(i: usize, j: usize, width: usize, height: usize, fov_deg: f32) -> Ray {
    todo!("C1: camera_ray")
}

/// C2a. The smallest positive root of `t² + 2 b t + k = 0`, or `None` if there is none.
/// Computed in the stable way: first the root of larger size, `q = −(b + sign(b) √(b² − k))`,
/// then the other one as `k / q` (the product of the roots is `k`).
///
/// Hint: if `b ≥ 0` then `q = −b − √(b² − k)`, else `q = −b + √(b² − k)`. The two roots are `q`
/// and `k / q`; return the smaller one if it is positive, else the larger one if that is
/// positive.
pub fn smaller_positive_root(b: f32, k: f32) -> Option<f32> {
    todo!("C2a: smaller_positive_root")
}

/// C2b. Where the ray (with a unit direction) first hits the sphere, or `None`.
///
/// Hint: `b = d · (o − c)`, `k = |o − c|² − r²`, then C2a. Vectors have `dot`, and `+`, `−`,
/// `*` with a number.
pub fn hit_sphere(ray: Ray, centre: Vec3, radius: f32) -> Option<f32> {
    todo!("C2b: hit_sphere")
}

/// C3. Where the ray hits the plane through `point` with normal `normal`, or `None` if it
/// runs parallel to the plane or meets it behind the eye.
///
/// Hint: `t = (p0 − o) · n / (d · n)`.
pub fn hit_plane(ray: Ray, point: Vec3, normal: Vec3) -> Option<f32> {
    todo!("C3: hit_plane")
}

/// C4. Ray and triangle `a, b, c`: solve `o + t d = a + u (b − a) + v (c − a)` by Cramer's
/// rule. Returns `(t, u, v)` for a hit (`u, v ≥ 0`, `u + v ≤ 1`, `t > 0`), else `None`.
/// The barycentric coordinates of the hit are `(1 − u − v, u, v)`.
///
/// Hint: the determinant of three columns is `x.dot(y.cross(z))`. Cramer: replace one column of
/// `[−d, e1, e2]` by `o − a` and divide by the determinant of the matrix.
pub fn hit_triangle(ray: Ray, a: Vec3, b: Vec3, c: Vec3) -> Option<(f32, f32, f32)> {
    todo!("C4: hit_triangle")
}

/// C5. Ray casting for one ray: test every object, keep the smallest positive `t`, and
/// give the normal at that hit, turned towards the eye.
///
/// Hint: `match` on the shape; the sphere's normal is `(p − c) / r`, the triangle's `(b − a) × (c − a)`,
/// normalised. Flip the normal if `normal · d > 0`.
pub fn nearest_hit(ray: Ray, scene: &[Shape]) -> Option<Hit> {
    todo!("C5: nearest_hit")
}
