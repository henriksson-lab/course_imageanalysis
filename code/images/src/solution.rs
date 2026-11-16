// Model solution for the coding exercises of "Images as arrays & geometric transforms".

use imgcourse::{Border, Image};

/// A 3 × 3 matrix in homogeneous coordinates, row by row: `m[row][column]`.
/// An affine transform has the last row `[0, 0, 1]`.
pub type Mat3 = [[f32; 3]; 3];

pub const IDENTITY: Mat3 = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];

/// C1a. Nearest-neighbour lookup at a position between pixels: the value of the closest
/// pixel, or 0 outside the image.
pub fn nearest(img: &Image<f32>, x: f32, y: f32) -> f32 {
    img.get_border(x.round() as isize, y.round() as isize, Border::Zero)
}

/// C1b. Bilinear interpolation: the weighted average of the 4 surrounding pixels
/// (pixels outside the image count as 0).
pub fn bilinear(img: &Image<f32>, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (a, b) = (x - x0, y - y0); // position inside the pixel square, 0..1
    let (x0, y0) = (x0 as isize, y0 as isize);
    let i = |dx: isize, dy: isize| img.get_border(x0 + dx, y0 + dy, Border::Zero);
    (1. - a) * (1. - b) * i(0, 0) + a * (1. - b) * i(1, 0) + (1. - a) * b * i(0, 1) + a * b * i(1, 1)
}

/// C2a. The matrix product `a b` (first apply `b`, then `a`).
pub fn mat3_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// C2b. Apply `m` to the point `(x, y)`: multiply `[x, y, 1]` by `m`.
pub fn apply(m: &Mat3, (x, y): (f32, f32)) -> (f32, f32) {
    (m[0][0] * x + m[0][1] * y + m[0][2], m[1][0] * x + m[1][1] * y + m[1][2])
}

/// C2c. Translation by `(tx, ty)`.
pub fn translation(tx: f32, ty: f32) -> Mat3 {
    [[1., 0., tx], [0., 1., ty], [0., 0., 1.]]
}

/// C2d. Rotation by `angle` (radians) about the origin.
pub fn rotation(angle: f32) -> Mat3 {
    let (s, c) = angle.sin_cos();
    [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
}

/// C2e. Rotation by `angle` about the point `(px, py)`: shift the point to the origin,
/// rotate, shift back.
pub fn rotate_about(px: f32, py: f32, angle: f32) -> Mat3 {
    mat3_mul(&translation(px, py), &mat3_mul(&rotation(angle), &translation(-px, -py)))
}

/// C3. The inverse of an affine transform, or `None` if it squashes the plane onto a line
/// (determinant 0).
pub fn invert_affine(m: &Mat3) -> Option<Mat3> {
    let [[a, b, tx], [c, d, ty], _] = *m;
    let det = a * d - b * c;
    if det.abs() < 1e-12 {
        return None;
    }
    // The inverse of the 2 × 2 part, then the shift that undoes t: x = A⁻¹ x' − A⁻¹ t.
    let (ia, ib, ic, id) = (d / det, -b / det, -c / det, a / det);
    Some([[ia, ib, -(ia * tx + ib * ty)], [ic, id, -(ic * tx + id * ty)], [0., 0., 1.]])
}

/// C4. Transform an image by the affine map `m` (input position -> output position),
/// computed backwards: every output pixel looks up where it came from with `interp`
/// (`nearest` or `bilinear`). The output has the same size as the input.
pub fn warp(img: &Image<f32>, m: &Mat3, interp: fn(&Image<f32>, f32, f32) -> f32) -> Image<f32> {
    let back = invert_affine(m).expect("the transform cannot be undone (determinant 0)");
    img.map_xy(|x, y, _| {
        let (sx, sy) = apply(&back, (x as f32, y as f32));
        interp(img, sx, sy)
    })
}
