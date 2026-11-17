// Coding exercises for "Non-rigid registration". Replace each `todo!()` with your code,
// then check it with `cargo test -p nonrigid-reg` and look at `cargo run -p nonrigid-reg`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};

/// C1. The cubic B-spline: `2/3 − t² + |t|³/2` for `|t| < 1`, `(2 − |t|)³/6` for
/// `1 ≤ |t| < 2`, and 0 beyond.
pub fn bspline3(t: f32) -> f32 {
    todo!("C1: bspline3")
}

/// C2. The displacement of a free-form deformation at `(x, y)`. The coefficients are two
/// small images `cx`, `cy` (one value per control point); control point `(k, l)` sits at the
/// pixel `((k − 1) Δ, (l − 1) Δ)`, so that one row and column of control points lie outside
/// the image on each side. `d(x, y) = Σ c_kl β(x/Δ − k + 1) β(y/Δ − l + 1)`: only the 4 × 4
/// control points with `k = ⌊x/Δ⌋ .. ⌊x/Δ⌋ + 3` (and likewise `l`) contribute.
///
/// Hint: `k0 = (x / spacing).floor() as isize`; skip control points outside the grid with
/// `cx.contains(k, l)`; the weight of control point (k, l) is
/// `bspline3(x / spacing - k as f32 + 1.0) * bspline3(y / spacing - l as f32 + 1.0)`.
pub fn ffd_displacement(cx: &Image<f32>, cy: &Image<f32>, spacing: f32, x: f32, y: f32) -> (f32, f32) {
    todo!("C2: ffd_displacement")
}

/// C3. The Jacobian determinant of `x ↦ x + d(x)` at every pixel, from the displacement
/// images `dx`, `dy` by central differences (edge copied outside):
/// `(1 + ∂dx/∂x)(1 + ∂dy/∂y) − (∂dx/∂y)(∂dy/∂x)`.
///
/// Hint: `∂dx/∂x ≈ (dx(x+1, y) − dx(x−1, y)) / 2`; read with `get_border(.., Border::Clamp)`.
pub fn jacobian_determinant(dx: &Image<f32>, dy: &Image<f32>) -> Image<f32> {
    todo!("C3: jacobian_determinant")
}

/// C4a. The entropy in bits of a histogram of probabilities (that add up to 1):
/// `H = −Σ p log2 p`, leaving out the empty bins.
///
/// Hint: `q.log2()`; skip `q = 0` (its term is 0).
pub fn entropy(p: &[f32]) -> f32 {
    todo!("C4a: entropy")
}

/// C4b. The mutual information in bits of two images of the same size, values in [0, 1]:
/// value `v` falls in bin `min(⌊v · bins⌋, bins − 1)` (values outside [0, 1] are clamped).
/// Joint histogram → probabilities → `H(A) + H(B) − H(A, B)`.
///
/// Hint: one `Vec` of `bins * bins` counts, index `i * bins + j`; divide by the number of
/// pixels; the two marginal histograms are its row and column sums.
pub fn mutual_information(a: &Image<f32>, b: &Image<f32>, bins: usize) -> f32 {
    todo!("C4b: mutual_information")
}

/// C5. Free-form registration by gradient descent on the SSD
/// `C = Σ (M(x + d(x)) − F(x))²`: start from zero coefficients (a grid of control points
/// every `spacing` px covering the image); in every iteration compute, for every pixel, `d`
/// (C2), the residual `r = M(x + d) − F(x)` and the gradient of the moving image at `x + d`;
/// add `2 r M_x β β` to the gradient of each of its 16 control points (and `2 r M_y β β` for
/// `cy`); then move every coefficient by `−step ×` its gradient. Returns `(cx, cy)`.
///
/// Hint: the gradient images of the moving image do not change: compute them once with
/// `gradient(moving)`, and read them at `x + d` with `bilinear`. The coefficient grid has
/// `ceil(width / spacing) + 3` columns (and likewise rows).
pub fn register_ssd(fixed: &Image<f32>, moving: &Image<f32>, spacing: f32, iterations: usize, step: f32) -> (Image<f32>, Image<f32>) {
    todo!("C5: register_ssd")
}

/// Not an exercise: the gradient `(∂I/∂x, ∂I/∂y)` by central differences, edge copied outside.
pub fn gradient(img: &Image<f32>) -> (Image<f32>, Image<f32>) {
    let p = |x: usize, y: usize, dx: isize, dy: isize| img.get_border(x as isize + dx, y as isize + dy, Border::Clamp);
    (img.map_xy(|x, y, _| (p(x, y, 1, 0) - p(x, y, -1, 0)) / 2.0), img.map_xy(|x, y, _| (p(x, y, 0, 1) - p(x, y, 0, -1)) / 2.0))
}

/// Not an exercise: bilinear interpolation, the edge copied outside the image.
pub fn bilinear(img: &Image<f32>, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let p = |i: f32, j: f32| img.get_border(i as isize, j as isize, Border::Clamp);
    (1.0 - fy) * ((1.0 - fx) * p(x0, y0) + fx * p(x0 + 1.0, y0)) + fy * ((1.0 - fx) * p(x0, y0 + 1.0) + fx * p(x0 + 1.0, y0 + 1.0))
}
