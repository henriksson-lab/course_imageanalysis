// Model solution for the coding exercises of "The Hessian".

use crate::gaussian_blur;
use imgcourse::{Border, Image};

/// C1a. The Hessian [Lxx, Lxy, Lyy] of `l` at pixel (x, y), from second differences
/// (copy the edge outside the image):
/// Lxx = f(1, 0) − 2 f(0, 0) + f(−1, 0), Lyy likewise along y, and
/// Lxy = ¼ [f(1, 1) − f(1, −1) − f(−1, 1) + f(−1, −1)], where f(dx, dy) = l(x + dx, y + dy).
pub fn hessian_at(l: &Image<f32>, x: usize, y: usize) -> [f32; 3] {
    let f = |dx: isize, dy: isize| l.get_border(x as isize + dx, y as isize + dy, Border::Clamp);
    let lxx = f(1, 0) - 2.0 * f(0, 0) + f(-1, 0);
    let lyy = f(0, 1) - 2.0 * f(0, 0) + f(0, -1);
    let lxy = 0.25 * (f(1, 1) - f(1, -1) - f(-1, 1) + f(-1, -1));
    [lxx, lxy, lyy]
}

/// C1b. The scale-normalised Hessian at every pixel: blur with a Gaussian of width `sigma`,
/// take `hessian_at`, and multiply all three entries by σ² (so that scales can be compared).
pub fn hessian(img: &Image<f32>, sigma: f32) -> Image<[f32; 3]> {
    let l = gaussian_blur(img, sigma);
    l.map_xy(|x, y, _| hessian_at(&l, x, y).map(|v| sigma * sigma * v))
}

/// C2. The eigenvalues of the symmetric matrix [[a, b], [b, c]] with h = [a, b, c], ordered by
/// size, |λ1| ≤ |λ2|, and the unit eigenvector of λ1 (the direction of the weaker curvature:
/// along a ridge). Returns (λ1, λ2, [ux, uy]).
pub fn eigen_sym(h: [f32; 3]) -> (f32, f32, [f32; 2]) {
    let [a, b, c] = h;
    let mean = 0.5 * (a + c);
    let root = (0.25 * (a - c) * (a - c) + b * b).sqrt();
    let (plus, minus) = (mean + root, mean - root);
    let (l1, l2) = if plus.abs() <= minus.abs() { (plus, minus) } else { (minus, plus) };
    // An eigenvector of λ solves (a − λ) u + b v = 0: take (b, λ − a), or (λ − c, b);
    // use the longer of the two, which is never (0, 0) unless the matrix is λ times I.
    let (p, q) = ([b, l1 - a], [l1 - c, b]);
    let u = if p[0].hypot(p[1]) >= q[0].hypot(q[1]) { p } else { q };
    let n = u[0].hypot(u[1]);
    let u = if n > 0.0 { [u[0] / n, u[1] / n] } else { [1.0, 0.0] };
    (l1, l2, u)
}

/// The structure at a pixel, read from its two eigenvalues.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Flat,
    BrightBlob,
    DarkBlob,
    BrightRidge,
    DarkRidge,
    Saddle,
}

/// C3. Classify a pixel from its eigenvalues (|λ1| ≤ |λ2|): flat if |λ2| < `flat`;
/// otherwise a ridge if |λ1| < ½ |λ2| (bright if λ2 < 0, dark if λ2 > 0); otherwise a blob if
/// both have the same sign (bright if negative), and a saddle if they have opposite signs.
pub fn classify(l1: f32, l2: f32, flat: f32) -> Class {
    if l2.abs() < flat {
        Class::Flat
    } else if l1.abs() < 0.5 * l2.abs() {
        if l2 < 0.0 {
            Class::BrightRidge
        } else {
            Class::DarkRidge
        }
    } else if l1 * l2 > 0.0 {
        if l2 < 0.0 {
            Class::BrightBlob
        } else {
            Class::DarkBlob
        }
    } else {
        Class::Saddle
    }
}

/// C4. Frangi's vesselness for bright lines: 0 if λ2 > 0, otherwise
/// exp(−R²/2β²) · (1 − exp(−S²/2c²)) with R = λ1/λ2 and S = √(λ1² + λ2²).
pub fn vesselness(l1: f32, l2: f32, beta: f32, c: f32) -> f32 {
    if l2 >= 0.0 {
        return 0.0;
    }
    let r = l1 / l2;
    let s2 = l1 * l1 + l2 * l2;
    (-r * r / (2.0 * beta * beta)).exp() * (1.0 - (-s2 / (2.0 * c * c)).exp())
}

/// C5. Multi-scale vesselness: at every pixel, the largest vesselness over the scales
/// `sigmas` (normalised Hessian), and the σ where it was largest (0 where all are 0).
/// For dark lines (`bright` false), flip the signs of both eigenvalues first.
pub fn multiscale_vesselness(img: &Image<f32>, sigmas: &[f32], beta: f32, c: f32, bright: bool) -> (Image<f32>, Image<f32>) {
    let mut best = Image::new(img.width(), img.height(), 0.0f32);
    let mut best_sigma = Image::new(img.width(), img.height(), 0.0f32);
    let sign = if bright { 1.0 } else { -1.0 };
    for &sigma in sigmas {
        let h = hessian(img, sigma);
        for (x, y) in img.coords() {
            let (l1, l2, _) = eigen_sym(h[(x, y)]);
            let v = vesselness(sign * l1, sign * l2, beta, c);
            if v > best[(x, y)] {
                best[(x, y)] = v;
                best_sigma[(x, y)] = sigma;
            }
        }
    }
    (best, best_sigma)
}
