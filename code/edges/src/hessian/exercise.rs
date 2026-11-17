// Coding exercises for "The Hessian". Replace each `todo!()` with your
// code, then check it with `cargo test -p edges hessian::` and look at `cargo run -p edges -- hessian`.
#![allow(unused_variables, unused_imports)]

use imgcourse::filters::gaussian_blur;
use imgcourse::{Border, Image};

/// C1a. The Hessian [Lxx, Lxy, Lyy] of `l` at pixel (x, y), from second differences
/// (copy the edge outside the image):
/// Lxx = f(1, 0) − 2 f(0, 0) + f(−1, 0), Lyy likewise along y, and
/// Lxy = ¼ [f(1, 1) − f(1, −1) − f(−1, 1) + f(−1, −1)], where f(dx, dy) = l(x + dx, y + dy).
///
/// Hint: a small closure `let f = |dx: isize, dy: isize| l.get_border(x as isize + dx,
/// y as isize + dy, Border::Clamp);` makes the formulas read like the slide.
pub fn hessian_at(l: &Image<f32>, x: usize, y: usize) -> [f32; 3] {
    todo!("C1a: hessian_at")
}

/// C1b. The scale-normalised Hessian at every pixel: blur with a Gaussian of width `sigma`,
/// take `hessian_at`, and multiply all three entries by σ² (so that scales can be compared).
///
/// Hint: `gaussian_blur(img, sigma)` (in `imgcourse::filters`) gives L; `l.map_xy(|x, y, _| ...)` loops
/// over the pixels; `[a, b, c].map(|v| ...)` scales the three entries.
pub fn hessian(img: &Image<f32>, sigma: f32) -> Image<[f32; 3]> {
    todo!("C1b: hessian")
}

/// C2. The eigenvalues of the symmetric matrix [[a, b], [b, c]] with h = [a, b, c], ordered by
/// size, |λ1| ≤ |λ2|, and the unit eigenvector of λ1 (the direction of the weaker curvature:
/// along a ridge). Returns (λ1, λ2, [ux, uy]).
///
/// Hint: λ = (a + c)/2 ± √(((a − c)/2)² + b²), then swap so that |λ1| ≤ |λ2|. An
/// eigenvector of λ1 is (b, λ1 − a) or (λ1 − c, b): take the longer one (one of them can be
/// (0, 0) when b = 0) and scale it to length 1.
pub fn eigen_sym(h: [f32; 3]) -> (f32, f32, [f32; 2]) {
    todo!("C2: eigen_sym")
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
    todo!("C3: classify")
}

/// C4. Frangi's vesselness for bright lines: 0 if λ2 > 0, otherwise
/// exp(−R²/2β²) · (1 − exp(−S²/2c²)) with R = λ1/λ2 and S = √(λ1² + λ2²).
///
/// Hint: `(-x).exp()` is e^(−x).
pub fn vesselness(l1: f32, l2: f32, beta: f32, c: f32) -> f32 {
    todo!("C4: vesselness")
}

/// C5. Multi-scale vesselness: at every pixel, the largest vesselness over the scales
/// `sigmas` (normalised Hessian), and the σ where it was largest (0 where all are 0).
/// For dark lines (`bright` false), flip the signs of both eigenvalues first.
///
/// Hint: start with two images of zeros; for each σ compute `hessian(img, sigma)` once, then
/// at every pixel compare the vesselness with the best so far.
pub fn multiscale_vesselness(img: &Image<f32>, sigmas: &[f32], beta: f32, c: f32, bright: bool) -> (Image<f32>, Image<f32>) {
    todo!("C5: multiscale_vesselness")
}
