// Model solution for the coding exercises of "Horn–Schunck".

use imgcourse::{Border, Image};

/// C1. One Horn–Schunck update of one pixel: from the neighbour average `(ubar, vbar)`, step
/// towards the pixel's constraint line:
/// `r = (Ix ubar + Iy vbar + It) / (4λ + Ix² + Iy²)`, `u = ubar − Ix r`, `v = vbar − Iy r`.
pub fn hs_update(ix: f32, iy: f32, it: f32, ubar: f32, vbar: f32, lambda: f32) -> (f32, f32) {
    let r = (ix * ubar + iy * vbar + it) / (4.0 * lambda + ix * ix + iy * iy);
    (ubar - ix * r, vbar - iy * r)
}

/// C2. The average of the four neighbours (left, right, up, down) of pixel `(x, y)`; outside
/// the image the edge pixel is copied.
pub fn neighbour_average(f: &Image<f32>, x: usize, y: usize) -> f32 {
    let p = |dx: isize, dy: isize| f.get_border(x as isize + dx, y as isize + dy, Border::Clamp);
    (p(-1, 0) + p(1, 0) + p(0, -1) + p(0, 1)) / 4.0
}

/// C3a. One sweep of Horn–Schunck: for every pixel, the neighbour averages of the **old**
/// field `(u, v)` and the update (C1). Returns the new field.
pub fn hs_sweep(ix: &Image<f32>, iy: &Image<f32>, it: &Image<f32>, u: &Image<f32>, v: &Image<f32>, lambda: f32) -> (Image<f32>, Image<f32>) {
    let (mut nu, mut nv) = (u.clone(), v.clone());
    for (x, y) in u.coords() {
        let (ubar, vbar) = (neighbour_average(u, x, y), neighbour_average(v, x, y));
        let (a, b) = hs_update(ix[(x, y)], iy[(x, y)], it[(x, y)], ubar, vbar, lambda);
        nu[(x, y)] = a;
        nv[(x, y)] = b;
    }
    (nu, nv)
}

/// C3b. Horn–Schunck: start from zero flow and do `sweeps` sweeps. Returns `(u, v)`.
pub fn horn_schunck(ix: &Image<f32>, iy: &Image<f32>, it: &Image<f32>, lambda: f32, sweeps: usize) -> (Image<f32>, Image<f32>) {
    let mut u = Image::new(ix.width(), ix.height(), 0.0);
    let mut v = u.clone();
    for _ in 0..sweeps {
        (u, v) = hs_sweep(ix, iy, it, &u, &v, lambda);
    }
    (u, v)
}

/// C4a. The endpoint error: the length of (estimate − truth), in pixels.
pub fn endpoint_error(est: (f32, f32), truth: (f32, f32)) -> f32 {
    (est.0 - truth.0).hypot(est.1 - truth.1)
}

/// C4b. The angular error in degrees: the angle between `(u, v, 1)` and `(u_gt, v_gt, 1)`.
pub fn angular_error(est: (f32, f32), truth: (f32, f32)) -> f32 {
    let dot = est.0 * truth.0 + est.1 * truth.1 + 1.0;
    let lengths = (est.0 * est.0 + est.1 * est.1 + 1.0).sqrt() * (truth.0 * truth.0 + truth.1 * truth.1 + 1.0).sqrt();
    (dot / lengths).clamp(-1.0, 1.0).acos().to_degrees()
}

/// C5. The warping residual: frame 2 read at `(x + u, y + v)` (with `bilinear`) minus frame 1.
pub fn warp_residual(f1: &Image<f32>, f2: &Image<f32>, u: &Image<f32>, v: &Image<f32>) -> Image<f32> {
    f1.map_xy(|x, y, a| bilinear(f2, x as f32 + u[(x, y)], y as f32 + v[(x, y)]) - a)
}

/// Not an exercise: `Ix`, `Iy` by central differences of the mean of the two frames, and
/// `It = f2 − f1` (as in the Lucas–Kanade unit).
pub fn gradients(f1: &Image<f32>, f2: &Image<f32>) -> (Image<f32>, Image<f32>, Image<f32>) {
    let mean = f1.zip_map(f2, |a, b| 0.5 * (a + b));
    let p = |x: usize, y: usize, dx: isize, dy: isize| mean.get_border(x as isize + dx, y as isize + dy, Border::Clamp);
    let ix = mean.map_xy(|x, y, _| (p(x, y, 1, 0) - p(x, y, -1, 0)) / 2.0);
    let iy = mean.map_xy(|x, y, _| (p(x, y, 0, 1) - p(x, y, 0, -1)) / 2.0);
    (ix, iy, f2.zip_map(f1, |a, b| a - b))
}

/// Not an exercise: bilinear interpolation, the edge copied outside the image.
pub fn bilinear(img: &Image<f32>, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let p = |i: f32, j: f32| img.get_border(i as isize, j as isize, Border::Clamp);
    (1.0 - fy) * ((1.0 - fx) * p(x0, y0) + fx * p(x0 + 1.0, y0)) + fy * ((1.0 - fx) * p(x0, y0 + 1.0) + fx * p(x0 + 1.0, y0 + 1.0))
}
