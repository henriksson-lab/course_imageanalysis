// Coding exercises for "Images as arrays & geometric transforms". Replace each `todo!()`
// with your code, then check it with `cargo test -p images images::` and look at `cargo run -p images -- images`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};

/// A 3 × 3 matrix in homogeneous coordinates, row by row: `m[row][column]`.
/// An affine transform has the last row `[0, 0, 1]`.
pub type Mat3 = [[f32; 3]; 3];

pub const IDENTITY: Mat3 = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];

/// C1a. Nearest-neighbour lookup at a position between pixels: the value of the closest
/// pixel, or 0 outside the image.
///
/// Hint: `img.get_border(x.round() as isize, y.round() as isize, Border::Zero)`.
pub fn nearest(img: &Image<f32>, x: f32, y: f32) -> f32 {
    todo!("C1a: nearest")
}

/// C1b. Bilinear interpolation: the weighted average of the 4 surrounding pixels
/// (pixels outside the image count as 0).
///
/// Hint: the pixel square has its top-left corner at `(x.floor(), y.floor())`; the
/// position inside it is `a = x - x.floor()`, `b = y - y.floor()`. Weights as on the slide.
pub fn bilinear(img: &Image<f32>, x: f32, y: f32) -> f32 {
    todo!("C1b: bilinear")
}

/// C2a. The matrix product `a b` (first apply `b`, then `a`).
pub fn mat3_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    todo!("C2a: mat3_mul")
}

/// C2b. Apply `m` to the point `(x, y)`: multiply `[x, y, 1]` by `m`.
pub fn apply(m: &Mat3, (x, y): (f32, f32)) -> (f32, f32) {
    todo!("C2b: apply")
}

/// C2c. Translation by `(tx, ty)`.
pub fn translation(tx: f32, ty: f32) -> Mat3 {
    todo!("C2c: translation")
}

/// C2d. Rotation by `angle` (radians) about the origin.
pub fn rotation(angle: f32) -> Mat3 {
    todo!("C2d: rotation")
}

/// C2e. Rotation by `angle` about the point `(px, py)`: shift the point to the origin,
/// rotate, shift back.
pub fn rotate_about(px: f32, py: f32, angle: f32) -> Mat3 {
    todo!("C2e: rotate_about")
}

/// C3. The inverse of an affine transform, or `None` if it squashes the plane onto a line
/// (determinant 0).
///
/// Hint: invert the 2 × 2 part `[[a, b], [c, d]]` (swap a and d, negate b and c, divide
/// by the determinant); the new shift is minus the inverse 2 × 2 part times the old shift.
pub fn invert_affine(m: &Mat3) -> Option<Mat3> {
    todo!("C3: invert_affine")
}

/// C4. Transform an image by the affine map `m` (input position -> output position),
/// computed backwards: every output pixel looks up where it came from with `interp`
/// (`nearest` or `bilinear`). The output has the same size as the input.
///
/// Hint: invert `m` once, then for every output pixel `(x, y)` apply the inverse to
/// `(x as f32, y as f32)` and call `interp(img, sx, sy)`. `img.map_xy` loops for you.
pub fn warp(img: &Image<f32>, m: &Mat3, interp: fn(&Image<f32>, f32, f32) -> f32) -> Image<f32> {
    todo!("C4: warp")
}
