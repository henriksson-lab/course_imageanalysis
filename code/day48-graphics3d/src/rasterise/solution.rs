// Model solution for the coding exercises of "From triangles to shaded pixels".
// Screen points are (x, y) in pixels with y up, as in the lecture; pixel (i, j) has its
// centre at (i + 0.5, j + 0.5). In the images, row j is drawn from the bottom.

use imgcourse::Image;

/// A point on the screen: (x, y).
pub type Vec2 = [f64; 2];
/// A direction in 3D.
pub type Vec3 = [f64; 3];

/// `dot` and `normalize` of 3D vectors, in `src/common.rs`.
pub use crate::common::{dot, normalize};

/// C1. The edge function `E_ab(p)`: positive if `p` lies to the left of the line from `a`
/// to `b` (slide "Which pixels does a triangle cover?"), twice the signed area of `a, b, p`.
pub fn edge(a: Vec2, b: Vec2, p: Vec2) -> f64 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

/// C2. The barycentric coordinates `(b_a, b_b, b_c)` of `p` in the triangle `a, b, c`, or
/// `None` if `p` is outside, or if the triangle is clockwise or flat (area ≤ 0: a back face,
/// culled). A point exactly on an edge counts as inside.
pub fn barycentric(a: Vec2, b: Vec2, c: Vec2, p: Vec2) -> Option<[f64; 3]> {
    let area = edge(a, b, c);
    if area <= 0.0 {
        return None;
    }
    let (ea, eb, ec) = (edge(b, c, p), edge(c, a, p), edge(a, b, p));
    if ea < 0.0 || eb < 0.0 || ec < 0.0 {
        return None;
    }
    Some([ea / area, eb / area, ec / area])
}

/// C3. Draw one triangle with a depth buffer. `v` holds the three corners as (x, y, depth).
/// For every pixel centre in the triangle's bounding box (clipped to the image): if it is
/// inside, interpolate the depth; if that is nearer (smaller) than `depth[(i, j)]`, store the
/// depth and set `colour[(i, j)] = shade(barycentrics)`. Returns the number of pixels drawn.
pub fn draw_triangle(colour: &mut Image<f32>, depth: &mut Image<f32>, v: [Vec3; 3], shade: &dyn Fn([f64; 3]) -> f32) -> usize {
    let (w, h) = (colour.width() as f64, colour.height() as f64);
    let (a, b, c) = ([v[0][0], v[0][1]], [v[1][0], v[1][1]], [v[2][0], v[2][1]]);
    // the bounding box, in whole pixels, inside the image
    let x0 = a[0].min(b[0]).min(c[0]).floor().max(0.0) as usize;
    let y0 = a[1].min(b[1]).min(c[1]).floor().max(0.0) as usize;
    let x1 = a[0].max(b[0]).max(c[0]).ceil().min(w) as usize;
    let y1 = a[1].max(b[1]).max(c[1]).ceil().min(h) as usize;
    let mut drawn = 0;
    for j in y0..y1 {
        for i in x0..x1 {
            let p = [i as f64 + 0.5, j as f64 + 0.5];
            if let Some(bary) = barycentric(a, b, c, p) {
                let z = (bary[0] * v[0][2] + bary[1] * v[1][2] + bary[2] * v[2][2]) as f32;
                if z < depth[(i, j)] {
                    depth[(i, j)] = z;
                    colour[(i, j)] = shade(bary);
                    drawn += 1;
                }
            }
        }
    }
    drawn
}

/// C4. Perspective-correct interpolation of a value `f` given at the three corners, with
/// screen barycentrics `b` and the corners' `w` (their distance in front of the camera):
/// interpolate `f / w` and `1 / w` on the screen, then divide.
pub fn perspective_correct(b: [f64; 3], w: [f64; 3], f: [f64; 3]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for k in 0..3 {
        num += b[k] * f[k] / w[k];
        den += b[k] / w[k];
    }
    num / den
}

/// C5a. Phong shading: `ka + kd max(0, n·l) + ks max(0, r·v)^s` with the mirror direction
/// `r = 2 (n·l) n − l`. All vectors have length 1 and point away from the surface.
pub fn phong(n: Vec3, l: Vec3, v: Vec3, ka: f64, kd: f64, ks: f64, s: f64) -> f64 {
    let nl = dot(n, l);
    let r = [2.0 * nl * n[0] - l[0], 2.0 * nl * n[1] - l[1], 2.0 * nl * n[2] - l[2]];
    let spec = if nl > 0.0 { dot(r, v).max(0.0).powf(s) } else { 0.0 };
    ka + kd * nl.max(0.0) + ks * spec
}

/// C5b. Blinn–Phong shading: as Phong, but the specular term is `ks max(0, n·h)^s` with the
/// half vector `h = (l + v) / |l + v|`.
pub fn blinn_phong(n: Vec3, l: Vec3, v: Vec3, ka: f64, kd: f64, ks: f64, s: f64) -> f64 {
    let nl = dot(n, l);
    let h = normalize([l[0] + v[0], l[1] + v[1], l[2] + v[2]]);
    let spec = if nl > 0.0 { dot(n, h).max(0.0).powf(s) } else { 0.0 };
    ka + kd * nl.max(0.0) + ks * spec
}
