// Model solution for the coding exercises of "Splines on images: interpolation and free-form deformation".

use imgcourse::{Border, Image};

/// The pole of the cubic B-spline prefilter, `z1 = √3 − 2`.
pub const Z1: f32 = -0.267_949_2;

/// The cubic B-spline β³ (from the B-spline unit).
pub fn cubic_bspline(x: f32) -> f32 {
    let a = x.abs();
    if a <= 1.0 {
        2.0 / 3.0 - a * a + 0.5 * a * a * a
    } else if a < 2.0 {
        (2.0 - a).powi(3) / 6.0
    } else {
        0.0
    }
}

/// C1. The prefilter along one row: coefficients `c` with `(c[k−1] + 4 c[k] + c[k+1]) / 6 = f[k]`,
/// where `c` outside the row is read with `Border::Mirror` (slide "The math: tensor product and
/// prefilter"). Two recursive passes with the pole `z1`, after a gain of 6:
/// left to right `c⁺[k] = 6 f[k] + z1 c⁺[k−1]`, right to left `c[k] = z1 (c[k+1] − c⁺[k])`.
pub fn prefilter_1d(f: &[f32]) -> Vec<f32> {
    let n = f.len();
    let z = Z1;
    // Start of the left-to-right pass: the infinite sum c⁺[0] = 6 Σ_j z^j f[−j] over the mirrored
    // row; z^j is below 1e-8 after 14 terms.
    let mut start = 0.0;
    let mut zj = 1.0;
    for j in 0..15isize {
        let k = Border::Mirror.index(-j, n).unwrap();
        start += zj * 6.0 * f[k];
        zj *= z;
    }
    let mut cplus = vec![0.0; n];
    cplus[0] = start;
    for k in 1..n {
        cplus[k] = 6.0 * f[k] + z * cplus[k - 1];
    }
    // Start of the right-to-left pass: the mirror makes c[n] = c[n−1].
    let mut c = vec![0.0; n];
    c[n - 1] = z / (z - 1.0) * cplus[n - 1];
    for k in (0..n - 1).rev() {
        c[k] = z * (c[k + 1] - cplus[k]);
    }
    c
}

/// C2. The B-spline coefficients of an image: `prefilter_1d` along every row, then along every
/// column of the result.
pub fn prefilter(img: &Image<f32>) -> Image<f32> {
    let (w, h) = (img.width(), img.height());
    let mut rows = Image::new(w, h, 0.0);
    for y in 0..h {
        let c = prefilter_1d(img.row(y));
        for x in 0..w {
            rows[(x, y)] = c[x];
        }
    }
    let mut out = Image::new(w, h, 0.0);
    for x in 0..w {
        let column: Vec<f32> = (0..h).map(|y| rows[(x, y)]).collect();
        let c = prefilter_1d(&column);
        for y in 0..h {
            out[(x, y)] = c[y];
        }
    }
    out
}

/// C3. The value of the tensor-product cubic B-spline with coefficients `coef` at a position
/// `(x, y)` between pixels: `sum over k, l of coef[k, l] β³(x − k) β³(y − l)`, over the 4 × 4
/// coefficients near `(x, y)`; outside the image the coefficients are read with `Border::Mirror`.
pub fn bspline_value(coef: &Image<f32>, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor() as isize, y.floor() as isize);
    let mut sum = 0.0;
    for l in y0 - 1..=y0 + 2 {
        let wy = cubic_bspline(y - l as f32);
        for k in x0 - 1..=x0 + 2 {
            sum += coef.get_border(k, l, Border::Mirror) * cubic_bspline(x - k as f32) * wy;
        }
    }
    sum
}

/// C4a. The cubic convolution kernel (a = −1/2): `1.5|s|³ − 2.5|s|² + 1` for `|s| ≤ 1`,
/// `−0.5|s|³ + 2.5|s|² − 4|s| + 2` for `1 < |s| < 2`, else 0.
pub fn cubic_convolution_kernel(s: f32) -> f32 {
    let a = s.abs();
    if a <= 1.0 {
        1.5 * a * a * a - 2.5 * a * a + 1.0
    } else if a < 2.0 {
        -0.5 * a * a * a + 2.5 * a * a - 4.0 * a + 2.0
    } else {
        0.0
    }
}

/// C4b. Cubic convolution interpolation at `(x, y)`: like C3, but on the **pixels** themselves
/// (no prefilter) with the kernel of C4a; outside the image `Border::Mirror`.
pub fn cubic_convolution(img: &Image<f32>, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor() as isize, y.floor() as isize);
    let mut sum = 0.0;
    for l in y0 - 1..=y0 + 2 {
        let wy = cubic_convolution_kernel(y - l as f32);
        for k in x0 - 1..=x0 + 2 {
            sum += img.get_border(k, l, Border::Mirror) * cubic_convolution_kernel(x - k as f32) * wy;
        }
    }
    sum
}

/// C5. The free-form deformation at pixel position `(x, y)`: `u = sum over k, l of
/// d[k, l] β³(x/h − k) β³(y/h − l)`, where control point `(k, l)` sits at `(k h, l h)` and has
/// the displacement `d[(k, l)]` (slide "The math: the FFD displacement field"). Control points
/// outside the grid count as 0.
pub fn ffd_displacement(d: &Image<[f32; 2]>, h: f32, x: f32, y: f32) -> [f32; 2] {
    let (gx, gy) = (x / h, y / h);
    let (k0, l0) = (gx.floor() as isize, gy.floor() as isize);
    let mut u = [0.0, 0.0];
    for l in l0 - 1..=l0 + 2 {
        for k in k0 - 1..=k0 + 2 {
            if d.contains(k, l) {
                let w = cubic_bspline(gx - k as f32) * cubic_bspline(gy - l as f32);
                let dk = d[(k as usize, l as usize)];
                u[0] += w * dk[0];
                u[1] += w * dk[1];
            }
        }
    }
    u
}
