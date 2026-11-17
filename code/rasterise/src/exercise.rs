// Coding exercises for "From triangles to shaded pixels".
// Replace each `todo!()` with your code, then check it with `cargo test -p rasterise`
// and look at `cargo run -p rasterise`.
// Screen points are (x, y) in pixels with y up, as in the lecture; pixel (i, j) has its
// centre at (i + 0.5, j + 0.5). In the images, row j is drawn from the bottom.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// A point on the screen: (x, y).
pub type Vec2 = [f64; 2];
/// A direction in 3D.
pub type Vec3 = [f64; 3];

pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn normalize(a: Vec3) -> Vec3 {
    let len = dot(a, a).sqrt();
    [a[0] / len, a[1] / len, a[2] / len]
}

/// C1. The edge function `E_ab(p)`: positive if `p` lies to the left of the line from `a`
/// to `b` (slide "Which pixels does a triangle cover?"), twice the signed area of `a, b, p`.
pub fn edge(a: Vec2, b: Vec2, p: Vec2) -> f64 {
    todo!("C1: edge")
}

/// C2. The barycentric coordinates `(b_a, b_b, b_c)` of `p` in the triangle `a, b, c`, or
/// `None` if `p` is outside, or if the triangle is clockwise or flat (area ≤ 0: a back face,
/// culled). A point exactly on an edge counts as inside.
///
/// Hint: compute the area `edge(a, b, c)` first and return `None` if it is ≤ 0. Then
/// `b_a = edge(b, c, p) / area`, `b_b = edge(c, a, p) / area`, `b_c = edge(a, b, p) / area`.
pub fn barycentric(a: Vec2, b: Vec2, c: Vec2, p: Vec2) -> Option<[f64; 3]> {
    todo!("C2: barycentric")
}

/// C3. Draw one triangle with a depth buffer. `v` holds the three corners as (x, y, depth).
/// For every pixel centre in the triangle's bounding box (clipped to the image): if it is
/// inside, interpolate the depth; if that is nearer (smaller) than `depth[(i, j)]`, store the
/// depth and set `colour[(i, j)] = shade(barycentrics)`. Returns the number of pixels drawn.
///
/// Hint: the bounding box runs from `floor` of the smallest to `ceil` of the largest
/// corner coordinate, clipped to `0..width` and `0..height`. Use `barycentric` for the inside test,
/// and call `shade(bary)` only for pixels that pass the depth test.
pub fn draw_triangle(colour: &mut Image<f32>, depth: &mut Image<f32>, v: [Vec3; 3], shade: &dyn Fn([f64; 3]) -> f32) -> usize {
    todo!("C3: draw_triangle")
}

/// C4. Perspective-correct interpolation of a value `f` given at the three corners, with
/// screen barycentrics `b` and the corners' `w` (their distance in front of the camera):
/// interpolate `f / w` and `1 / w` on the screen, then divide.
///
/// Hint: two sums over the corners: `b[k] * f[k] / w[k]` and `b[k] / w[k]`.
pub fn perspective_correct(b: [f64; 3], w: [f64; 3], f: [f64; 3]) -> f64 {
    todo!("C4: perspective_correct")
}

/// C5a. Phong shading: `ka + kd max(0, n·l) + ks max(0, r·v)^s` with the mirror direction
/// `r = 2 (n·l) n − l`. All vectors have length 1 and point away from the surface.
///
/// Hint: compute the specular part only when `n·l > 0` (the light is in front of the surface).
pub fn phong(n: Vec3, l: Vec3, v: Vec3, ka: f64, kd: f64, ks: f64, s: f64) -> f64 {
    todo!("C5a: phong")
}

/// C5b. Blinn–Phong shading: as Phong, but the specular term is `ks max(0, n·h)^s` with the
/// half vector `h = (l + v) / |l + v|`.
///
/// Hint: `normalize` is given above.
pub fn blinn_phong(n: Vec3, l: Vec3, v: Vec3, ka: f64, kd: f64, ks: f64, s: f64) -> f64 {
    todo!("C5b: blinn_phong")
}
