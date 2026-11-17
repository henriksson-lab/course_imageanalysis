// Model solution for the coding exercises of "Non-rigid registration".

use imgcourse::{Border, Image};

/// C1. The cubic B-spline: `2/3 − t² + |t|³/2` for `|t| < 1`, `(2 − |t|)³/6` for
/// `1 ≤ |t| < 2`, and 0 beyond.
pub fn bspline3(t: f32) -> f32 {
    let a = t.abs();
    if a < 1.0 {
        2.0 / 3.0 - a * a + 0.5 * a * a * a
    } else if a < 2.0 {
        (2.0 - a).powi(3) / 6.0
    } else {
        0.0
    }
}

/// C2. The displacement of a free-form deformation at `(x, y)`. The coefficients are two
/// small images `cx`, `cy` (one value per control point); control point `(k, l)` sits at the
/// pixel `((k − 1) Δ, (l − 1) Δ)`, so that one row and column of control points lie outside
/// the image on each side. `d(x, y) = Σ c_kl β(x/Δ − k + 1) β(y/Δ − l + 1)`: only the 4 × 4
/// control points with `k = ⌊x/Δ⌋ .. ⌊x/Δ⌋ + 3` (and likewise `l`) contribute.
pub fn ffd_displacement(cx: &Image<f32>, cy: &Image<f32>, spacing: f32, x: f32, y: f32) -> (f32, f32) {
    let (k0, l0) = ((x / spacing).floor() as isize, (y / spacing).floor() as isize);
    let (mut dx, mut dy) = (0.0, 0.0);
    for l in l0..l0 + 4 {
        for k in k0..k0 + 4 {
            if !cx.contains(k, l) {
                continue;
            }
            let w = bspline3(x / spacing - k as f32 + 1.0) * bspline3(y / spacing - l as f32 + 1.0);
            dx += w * cx[(k as usize, l as usize)];
            dy += w * cy[(k as usize, l as usize)];
        }
    }
    (dx, dy)
}

/// C3. The Jacobian determinant of `x ↦ x + d(x)` at every pixel, from the displacement
/// images `dx`, `dy` by central differences (edge copied outside):
/// `(1 + ∂dx/∂x)(1 + ∂dy/∂y) − (∂dx/∂y)(∂dy/∂x)`.
pub fn jacobian_determinant(dx: &Image<f32>, dy: &Image<f32>) -> Image<f32> {
    dx.map_xy(|x, y, _| {
        let d = |f: &Image<f32>, ox: isize, oy: isize| {
            (f.get_border(x as isize + ox, y as isize + oy, Border::Clamp) - f.get_border(x as isize - ox, y as isize - oy, Border::Clamp)) / 2.0
        };
        (1.0 + d(dx, 1, 0)) * (1.0 + d(dy, 0, 1)) - d(dx, 0, 1) * d(dy, 1, 0)
    })
}

/// C4a. The entropy in bits of a histogram of probabilities (that add up to 1):
/// `H = −Σ p log2 p`, leaving out the empty bins.
pub fn entropy(p: &[f32]) -> f32 {
    p.iter().filter(|&&q| q > 0.0).map(|&q| -q * q.log2()).sum()
}

/// C4b. The mutual information in bits of two images of the same size, values in [0, 1]:
/// value `v` falls in bin `min(⌊v · bins⌋, bins − 1)` (values outside [0, 1] are clamped).
/// Joint histogram → probabilities → `H(A) + H(B) − H(A, B)`.
pub fn mutual_information(a: &Image<f32>, b: &Image<f32>, bins: usize) -> f32 {
    let bin = |v: f32| ((v.clamp(0.0, 1.0) * bins as f32) as usize).min(bins - 1);
    let mut joint = vec![0.0f32; bins * bins];
    for (&p, &q) in a.pixels().iter().zip(b.pixels()) {
        joint[bin(p) * bins + bin(q)] += 1.0;
    }
    let n = a.pixels().len() as f32;
    let joint: Vec<f32> = joint.iter().map(|c| c / n).collect();
    let mut pa = vec![0.0; bins];
    let mut pb = vec![0.0; bins];
    for i in 0..bins {
        for j in 0..bins {
            pa[i] += joint[i * bins + j];
            pb[j] += joint[i * bins + j];
        }
    }
    entropy(&pa) + entropy(&pb) - entropy(&joint)
}

/// C5. Free-form registration by gradient descent on the SSD
/// `C = Σ (M(x + d(x)) − F(x))²`: start from zero coefficients (a grid of control points
/// every `spacing` px covering the image); in every iteration compute, for every pixel, `d`
/// (C2), the residual `r = M(x + d) − F(x)` and the gradient of the moving image at `x + d`;
/// add `2 r M_x β β` to the gradient of each of its 16 control points (and `2 r M_y β β` for
/// `cy`); then move every coefficient by `−step ×` its gradient. Returns `(cx, cy)`.
pub fn register_ssd(fixed: &Image<f32>, moving: &Image<f32>, spacing: f32, iterations: usize, step: f32) -> (Image<f32>, Image<f32>) {
    let (gx, gy) = gradient(moving);
    let kw = (fixed.width() as f32 / spacing).ceil() as usize + 3;
    let kh = (fixed.height() as f32 / spacing).ceil() as usize + 3;
    let (mut cx, mut cy) = (Image::new(kw, kh, 0.0), Image::new(kw, kh, 0.0));
    for _ in 0..iterations {
        let (mut grad_x, mut grad_y) = (Image::new(kw, kh, 0.0), Image::new(kw, kh, 0.0));
        for (x, y) in fixed.coords() {
            let (xf, yf) = (x as f32, y as f32);
            let (dx, dy) = ffd_displacement(&cx, &cy, spacing, xf, yf);
            let r = bilinear(moving, xf + dx, yf + dy) - fixed[(x, y)];
            let (mx, my) = (bilinear(&gx, xf + dx, yf + dy), bilinear(&gy, xf + dx, yf + dy));
            // The same 4 × 4 control points as in ffd_displacement.
            let (k0, l0) = ((xf / spacing).floor() as usize, (yf / spacing).floor() as usize);
            for l in l0..(l0 + 4).min(kh) {
                for k in k0..(k0 + 4).min(kw) {
                    let w = bspline3(xf / spacing - k as f32 + 1.0) * bspline3(yf / spacing - l as f32 + 1.0);
                    grad_x[(k, l)] += 2.0 * r * mx * w;
                    grad_y[(k, l)] += 2.0 * r * my * w;
                }
            }
        }
        cx = cx.zip_map(&grad_x, |c, g| c - step * g);
        cy = cy.zip_map(&grad_y, |c, g| c - step * g);
    }
    (cx, cy)
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
