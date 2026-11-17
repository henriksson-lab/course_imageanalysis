//! Helpers shared by the units of the day: vectors in 3D.

/// A point or direction in 3D.
pub type Vec3 = [f64; 3];

pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn normalize(a: Vec3) -> Vec3 {
    let len = dot(a, a).sqrt();
    [a[0] / len, a[1] / len, a[2] / len]
}
