// Model solution for the coding exercises of "Rays from the camera, and where they hit".

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
pub fn camera_ray(i: usize, j: usize, width: usize, height: usize, fov_deg: f32) -> Ray {
    // Slide "The math: the ray through a pixel": normalised coordinates in [-1, 1].
    let xn = 2.0 * (i as f32 + 0.5) / width as f32 - 1.0;
    let yn = 1.0 - 2.0 * (j as f32 + 0.5) / height as f32;
    let tan_half = (0.5 * fov_deg).to_radians().tan();
    let aspect = width as f32 / height as f32;
    let d = Vec3 { x: xn * tan_half * aspect, y: yn * tan_half, z: -1.0 };
    Ray::new(Vec3::default(), d.normalize())
}

/// C2a. The smallest positive root of `t² + 2 b t + k = 0`, or `None` if there is none.
/// Computed in the stable way: first the root of larger size, `q = −(b + sign(b) √(b² − k))`,
/// then the other one as `k / q` (the product of the roots is `k`).
pub fn smaller_positive_root(b: f32, k: f32) -> Option<f32> {
    let disc = b * b - k;
    if disc < 0.0 {
        return None; // the ray misses
    }
    let s = disc.sqrt();
    let q = if b >= 0.0 { -b - s } else { -b + s };
    if q == 0.0 {
        return None; // both roots are 0
    }
    let (t1, t2) = (q, k / q);
    let (small, large) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
    if small > 0.0 {
        Some(small)
    } else if large > 0.0 {
        Some(large) // the eye is inside the sphere
    } else {
        None // behind the eye
    }
}

/// C2b. Where the ray (with a unit direction) first hits the sphere, or `None`.
pub fn hit_sphere(ray: Ray, centre: Vec3, radius: f32) -> Option<f32> {
    // Slide "The math: the sphere quadratic": b' = d·(o − c), k = |o − c|² − r².
    let oc = ray.origin - centre;
    let b = ray.dir.dot(oc);
    let k = oc.dot(oc) - radius * radius;
    smaller_positive_root(b, k)
}

/// C3. Where the ray hits the plane through `point` with normal `normal`, or `None` if it
/// runs parallel to the plane or meets it behind the eye.
pub fn hit_plane(ray: Ray, point: Vec3, normal: Vec3) -> Option<f32> {
    let denom = ray.dir.dot(normal);
    if denom == 0.0 {
        return None; // parallel
    }
    let t = (point - ray.origin).dot(normal) / denom;
    if t > 0.0 {
        Some(t)
    } else {
        None
    }
}

/// C4. Ray and triangle `a, b, c`: solve `o + t d = a + u (b − a) + v (c − a)` by Cramer's
/// rule. Returns `(t, u, v)` for a hit (`u, v ≥ 0`, `u + v ≤ 1`, `t > 0`), else `None`.
/// The barycentric coordinates of the hit are `(1 − u − v, u, v)`.
pub fn hit_triangle(ray: Ray, a: Vec3, b: Vec3, c: Vec3) -> Option<(f32, f32, f32)> {
    // Columns of the matrix: −d, e1, e2; right-hand side o − a.
    let (m0, e1, e2, rhs) = (-ray.dir, b - a, c - a, ray.origin - a);
    // det[x y z] = x · (y × z), the signed volume.
    let det = |x: Vec3, y: Vec3, z: Vec3| x.dot(y.cross(z));
    let d = det(m0, e1, e2);
    if d.abs() < 1e-12 {
        return None; // parallel to the triangle's plane
    }
    let t = det(rhs, e1, e2) / d;
    let u = det(m0, rhs, e2) / d;
    let v = det(m0, e1, rhs) / d;
    if u >= 0.0 && v >= 0.0 && u + v <= 1.0 && t > 0.0 {
        Some((t, u, v))
    } else {
        None
    }
}

/// C5. Ray casting for one ray: test every object, keep the smallest positive `t`, and
/// give the normal at that hit, turned towards the eye.
pub fn nearest_hit(ray: Ray, scene: &[Shape]) -> Option<Hit> {
    let mut best: Option<Hit> = None;
    let mut best_t = f32::INFINITY;
    for (index, shape) in scene.iter().enumerate() {
        let found = match *shape {
            Shape::Sphere { centre, radius } => hit_sphere(ray, centre, radius).map(|t| (t, (ray.at(t) - centre) * (1.0 / radius))),
            Shape::Plane { point, normal } => hit_plane(ray, point, normal).map(|t| (t, normal)),
            Shape::Triangle { a, b, c } => hit_triangle(ray, a, b, c).map(|(t, _, _)| (t, (b - a).cross(c - a).normalize())),
        };
        if let Some((t, normal)) = found {
            if t < best_t {
                // Seen from the back: flip the normal towards the eye.
                let normal = if normal.dot(ray.dir) > 0.0 { -normal } else { normal };
                best = Some(Hit { t, normal, object: index });
                best_t = t;
            }
        }
    }
    best
}
