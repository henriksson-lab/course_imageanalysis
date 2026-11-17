//! Helpers shared by the units of the day: 3-vectors and 3 × 3 matrices.

pub type Vec3 = [f64; 3];
/// A 3 × 3 matrix, row by row: `m[row][column]`.
pub type Mat3 = [[f64; 3]; 3];

pub fn mat_vec(m: &Mat3, v: Vec3) -> Vec3 {
    [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

pub fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    [0, 1, 2].map(|i| [0, 1, 2].map(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

pub fn transpose(m: &Mat3) -> Mat3 {
    [0, 1, 2].map(|i| [0, 1, 2].map(|j| m[j][i]))
}

pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
