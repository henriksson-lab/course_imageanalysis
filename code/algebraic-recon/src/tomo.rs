//! Plumbing for the tomography units (not an exercise): the coordinates of the lecture, a
//! test phantom, reading an image between pixels, showing a sinogram, and the projector of
//! the radon unit (your code there).
//!
//! Coordinates as in the lecture: the origin is the centre of the image, x to the right,
//! **y up**, angles counter-clockwise. Pixel `(px, py)` of an n × n image has
//! x = px − c and y = c − py, with c = (n − 1) / 2.

use imgcourse::Image;
use std::f32::consts::PI;

/// The centre c = (n − 1) / 2 of an image or detector with n pixels or bins.
pub fn centre(n: usize) -> f32 {
    (n as f32 - 1.0) / 2.0
}

/// `count` angles evenly spaced in [0°, 180°), in radians.
pub fn angles(count: usize) -> Vec<f32> {
    (0..count).map(|k| k as f32 * PI / count as f32).collect()
}

/// `count` angles evenly spaced from `from_deg` to `to_deg` (both included), in radians.
pub fn angles_between(from_deg: f32, to_deg: f32, count: usize) -> Vec<f32> {
    (0..count).map(|k| (from_deg + (to_deg - from_deg) * k as f32 / (count - 1) as f32).to_radians()).collect()
}

/// The value at the lecture position (x, y), between pixels: bilinear interpolation of the
/// four nearest pixels, 0 outside the image.
pub fn sample(img: &Image<f32>, x: f32, y: f32) -> f32 {
    let c = centre(img.width());
    let (fx, fy) = (x + c, c - y); // pixel coordinates
    let (x0, y0) = (fx.floor(), fy.floor());
    let (tx, ty) = (fx - x0, fy - y0);
    let get = |i: f32, j: f32| {
        if img.contains(i as isize, j as isize) {
            img[(i as usize, j as usize)]
        } else {
            0.0
        }
    };
    (1.0 - tx) * (1.0 - ty) * get(x0, y0) + tx * (1.0 - ty) * get(x0 + 1.0, y0) + (1.0 - tx) * ty * get(x0, y0 + 1.0) + tx * ty * get(x0 + 1.0, y0 + 1.0)
}

/// The (modified) Shepp–Logan phantom, n × n: ten ellipses that imitate a head section, values
/// between 0 and 1 (skull 1, brain 0.2, small structures 0.3).
pub fn shepp_logan(n: usize) -> Image<f32> {
    // value, half-axes a and b, centre (x0, y0), angle in degrees; the image spans [−1, 1]
    let ellipses: [(f32, f32, f32, f32, f32, f32); 10] = [
        (1.0, 0.69, 0.92, 0.0, 0.0, 0.0),
        (-0.8, 0.6624, 0.874, 0.0, -0.0184, 0.0),
        (-0.2, 0.11, 0.31, 0.22, 0.0, -18.0),
        (-0.2, 0.16, 0.41, -0.22, 0.0, 18.0),
        (0.1, 0.21, 0.25, 0.0, 0.35, 0.0),
        (0.1, 0.046, 0.046, 0.0, 0.1, 0.0),
        (0.1, 0.046, 0.046, 0.0, -0.1, 0.0),
        (0.1, 0.046, 0.023, -0.08, -0.605, 0.0),
        (0.1, 0.023, 0.023, 0.0, -0.606, 0.0),
        (0.1, 0.023, 0.046, 0.06, -0.605, 0.0),
    ];
    let c = centre(n);
    let scale = 2.0 / n as f32;
    Image::from_fn(n, n, |px, py| {
        let (x, y) = ((px as f32 - c) * scale, (c - py as f32) * scale);
        let mut v = 0.0;
        for (value, a, b, x0, y0, deg) in ellipses {
            let (s, co) = deg.to_radians().sin_cos();
            let (dx, dy) = (x - x0, y - y0);
            let (u, w) = (co * dx + s * dy, -s * dx + co * dy);
            if (u / a).powi(2) + (w / b).powi(2) <= 1.0 {
                v += value;
            }
        }
        v
    })
}

/// A sinogram as a picture the way the lecture draws it: θ across, ρ **up** (the sinogram
/// stores bin k in row k, so the picture is turned upside down), stretched to black–white.
pub fn sinogram_picture(sino: &Image<f32>) -> Image<f32> {
    let h = sino.height();
    let (lo, hi) = (sino.min(), sino.max());
    let range = if hi > lo { hi - lo } else { 1.0 };
    Image::from_fn(sino.width(), h, |a, y| (sino[(a, h - 1 - y)] - lo) / range)
}

/// The projection of an n × n image at angle θ (the radon unit's C2): n detector bins, bin k
/// at ρ = k − c; each bin adds up the image along its ray in steps of 1.
pub fn projection(img: &Image<f32>, theta: f32) -> Vec<f32> {
    let n = img.width();
    let c = centre(n);
    let (cos, sin) = (theta.cos(), theta.sin());
    (0..n)
        .map(|k| {
            let r = k as f32 - c;
            (0..n).map(|j| j as f32 - c).map(|s| sample(img, r * cos - s * sin, r * sin + s * cos)).sum()
        })
        .collect()
}

/// The sinogram (the radon unit's C3): `sino[(a, k)]` = bin k of the projection at `thetas[a]`.
pub fn sinogram(img: &Image<f32>, thetas: &[f32]) -> Image<f32> {
    let mut sino = Image::new(thetas.len(), img.width(), 0.0);
    for (a, &theta) in thetas.iter().enumerate() {
        for (k, v) in projection(img, theta).into_iter().enumerate() {
            sino[(a, k)] = v;
        }
    }
    sino
}

/// Back-projection (the back-projection unit's C2): every pixel adds, for every angle, the
/// value of the sinogram at its own ρ = x cos θ + y sin θ (read between bins), times π / N_θ.
pub fn backproject(sino: &Image<f32>, thetas: &[f32], n: usize) -> Image<f32> {
    let c = centre(n);
    let bins = sino.height();
    let cb = centre(bins);
    Image::from_fn(n, n, |px, py| {
        let (x, y) = (px as f32 - c, c - py as f32);
        let mut sum = 0.0;
        for (a, &theta) in thetas.iter().enumerate() {
            let t = x * theta.cos() + y * theta.sin() + cb; // the bin position, between bins
            let k = t.floor();
            let f = t - k;
            let k = k as isize;
            let get = |k: isize| if k >= 0 && (k as usize) < bins { sino[(a, k as usize)] } else { 0.0 };
            sum += (1.0 - f) * get(k) + f * get(k + 1);
        }
        sum * PI / thetas.len() as f32
    })
}

/// One row of the matrix A: the pixels a ray touches, as (pixel index y·n + x, weight).
pub type Row = Vec<(usize, f32)>;

/// The projector as a sparse matrix A, one row per ray (angle by angle, bin by bin): the same
/// rays and weights as `projection`, so that `A x` is the sinogram of the image `x` (pixels
/// row by row). The weights are those of `sample` (bilinear), added up per pixel.
pub fn ray_matrix(n: usize, thetas: &[f32]) -> Vec<Row> {
    let c = centre(n);
    let mut rows = Vec::new();
    for &theta in thetas {
        let (cos, sin) = (theta.cos(), theta.sin());
        for k in 0..n {
            let r = k as f32 - c;
            let mut weights: Vec<f32> = Vec::new();
            let mut pixels: Vec<usize> = Vec::new();
            for j in 0..n {
                let s = j as f32 - c;
                let (x, y) = (r * cos - s * sin, r * sin + s * cos);
                let (fx, fy) = (x + c, c - y);
                let (x0, y0) = (fx.floor(), fy.floor());
                let (tx, ty) = (fx - x0, fy - y0);
                for (dx, dy, w) in [(0.0, 0.0, (1.0 - tx) * (1.0 - ty)), (1.0, 0.0, tx * (1.0 - ty)), (0.0, 1.0, (1.0 - tx) * ty), (1.0, 1.0, tx * ty)] {
                    let (px, py) = (x0 + dx, y0 + dy);
                    if px >= 0.0 && py >= 0.0 && (px as usize) < n && (py as usize) < n && w > 0.0 {
                        pixels.push(py as usize * n + px as usize);
                        weights.push(w);
                    }
                }
            }
            // add up the weights of the same pixel
            let mut order: Vec<usize> = (0..pixels.len()).collect();
            order.sort_by_key(|&i| pixels[i]);
            let mut row: Row = Vec::new();
            for i in order {
                match row.last_mut() {
                    Some(last) if last.0 == pixels[i] => last.1 += weights[i],
                    _ => row.push((pixels[i], weights[i])),
                }
            }
            rows.push(row);
        }
    }
    rows
}
