// Model solution for the coding exercises of "Lucas–Kanade".

use imgcourse::{Border, Image};

pub use crate::common::bilinear;

/// C1. The derivatives of a frame pair: `Ix`, `Iy` by central differences of the **mean** of
/// the two frames (copy the edge outside), and `It = f2 − f1`.
pub fn gradients(f1: &Image<f32>, f2: &Image<f32>) -> (Image<f32>, Image<f32>, Image<f32>) {
    let mean = f1.zip_map(f2, |a, b| 0.5 * (a + b));
    let p = |x: usize, y: usize, dx: isize, dy: isize| mean.get_border(x as isize + dx, y as isize + dy, Border::Clamp);
    let ix = mean.map_xy(|x, y, _| (p(x, y, 1, 0) - p(x, y, -1, 0)) / 2.0);
    let iy = mean.map_xy(|x, y, _| (p(x, y, 0, 1) - p(x, y, 0, -1)) / 2.0);
    let it = f2.zip_map(f1, |a, b| a - b);
    (ix, iy, it)
}

/// C2a. Solve the normal equations `J (u, v) = −b` of one window, with `J` given as
/// `[Σ Ix², Σ Ix Iy, Σ Iy²]` and `b = [Σ Ix It, Σ Iy It]`. `None` if `|det J| < 1e-12`.
pub fn lk_solve(j: [f32; 3], b: [f32; 2]) -> Option<(f32, f32)> {
    let det = j[0] * j[2] - j[1] * j[1];
    if det.abs() < 1e-12 {
        return None;
    }
    // The inverse of [[a, c], [c, d]] is [[d, −c], [−c, a]] / det.
    let u = -(j[2] * b[0] - j[1] * b[1]) / det;
    let v = -(-j[1] * b[0] + j[0] * b[1]) / det;
    Some((u, v))
}

/// C2b. Lucas–Kanade for one window given as a list of pixels `[Ix, Iy, It]`: add up `J`
/// and `b`, then `lk_solve`.
pub fn lk_window(pixels: &[[f32; 3]]) -> Option<(f32, f32)> {
    let (mut j, mut b) = ([0.0; 3], [0.0; 2]);
    for &[ix, iy, it] in pixels {
        j[0] += ix * ix;
        j[1] += ix * iy;
        j[2] += iy * iy;
        b[0] += ix * it;
        b[1] += iy * it;
    }
    lk_solve(j, b)
}

/// C3. The smaller eigenvalue μ2 of `J = [[a, c], [c, d]]`, given as `[a, c, d]`: the confidence.
pub fn min_eigenvalue(j: [f32; 3]) -> f32 {
    let (a, c, d) = (j[0], j[1], j[2]);
    (a + d) / 2.0 - (((a - d) / 2.0).powi(2) + c * c).sqrt()
}

/// C4. Lucas–Kanade at every pixel, with a Gaussian window of width `rho`: the five sums of
/// `J` and `b` are five Gaussian blurs of products (`gaussian_blur`), then one solve per pixel.
/// Returns `(u, v, mu2)`; where the solve fails, the flow is 0.
pub fn lucas_kanade(f1: &Image<f32>, f2: &Image<f32>, rho: f32) -> (Image<f32>, Image<f32>, Image<f32>) {
    let (ix, iy, it) = gradients(f1, f2);
    let sxx = gaussian_blur(&ix.zip_map(&ix, |a, b| a * b), rho);
    let sxy = gaussian_blur(&ix.zip_map(&iy, |a, b| a * b), rho);
    let syy = gaussian_blur(&iy.zip_map(&iy, |a, b| a * b), rho);
    let sxt = gaussian_blur(&ix.zip_map(&it, |a, b| a * b), rho);
    let syt = gaussian_blur(&iy.zip_map(&it, |a, b| a * b), rho);
    let (w, h) = (f1.width(), f1.height());
    let (mut u, mut v, mut mu2) = (Image::new(w, h, 0.0), Image::new(w, h, 0.0), Image::new(w, h, 0.0));
    for (x, y) in f1.coords() {
        let j = [sxx[(x, y)], sxy[(x, y)], syy[(x, y)]];
        if let Some((a, b)) = lk_solve(j, [sxt[(x, y)], syt[(x, y)]]) {
            u[(x, y)] = a;
            v[(x, y)] = b;
        }
        mu2[(x, y)] = min_eigenvalue(j);
    }
    (u, v, mu2)
}

/// C5. Coarse to fine with warping: build pyramids of both frames with `levels` levels
/// (`downsample`), start with zero flow at the coarsest level; on every level (coarsest
/// first), `iterations` times: warp frame 2 back by the current flow (`warp`), estimate the
/// remaining flow with `lucas_kanade`, add it. Going one level finer: `upsample_flow`.
pub fn pyramid_lk(f1: &Image<f32>, f2: &Image<f32>, levels: usize, iterations: usize, rho: f32) -> (Image<f32>, Image<f32>) {
    let (mut p1, mut p2) = (vec![f1.clone()], vec![f2.clone()]);
    for l in 1..levels {
        p1.push(downsample(&p1[l - 1]));
        p2.push(downsample(&p2[l - 1]));
    }
    let top = levels - 1;
    let mut u = Image::new(p1[top].width(), p1[top].height(), 0.0);
    let mut v = u.clone();
    for l in (0..levels).rev() {
        if l < top {
            // One level finer: twice as many pixels, so the flow doubles.
            u = upsample_flow(&u, p1[l].width(), p1[l].height());
            v = upsample_flow(&v, p1[l].width(), p1[l].height());
        }
        for _ in 0..iterations {
            let warped = warp(&p2[l], &u, &v);
            let (du, dv, _) = lucas_kanade(&p1[l], &warped, rho);
            u = u.zip_map(&du, |a, b| a + b);
            v = v.zip_map(&dv, |a, b| a + b);
        }
    }
    (u, v)
}

/// Not an exercise: a Gaussian blur (two passes of a sampled, normalised 1D Gaussian of
/// radius 3σ, mirror border).
pub fn gaussian_blur(img: &Image<f32>, sigma: f32) -> Image<f32> {
    let r = (3.0 * sigma).ceil() as isize;
    let w: Vec<f32> = (-r..=r).map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let total: f32 = w.iter().sum();
    let pass = |src: &Image<f32>, along_x: bool| {
        src.map_xy(|x, y, _| {
            let mut sum = 0.0;
            for d in -r..=r {
                let (xx, yy) = if along_x { (x as isize + d, y as isize) } else { (x as isize, y as isize + d) };
                sum += w[(d + r) as usize] * src.get_border(xx, yy, Border::Mirror);
            }
            sum / total
        })
    };
    pass(&pass(img, true), false)
}

/// Not an exercise: one level up the pyramid: blur with σ = 1, keep every second pixel.
pub fn downsample(img: &Image<f32>) -> Image<f32> {
    let b = gaussian_blur(img, 1.0);
    Image::from_fn(img.width().div_ceil(2), img.height().div_ceil(2), |x, y| b[(2 * x, 2 * y)])
}

/// Not an exercise: a flow component one level down the pyramid: `w × h` pixels, bilinear,
/// and every value doubled (the pixels are half as large).
pub fn upsample_flow(f: &Image<f32>, w: usize, h: usize) -> Image<f32> {
    Image::from_fn(w, h, |x, y| 2.0 * bilinear(f, x as f32 / 2.0, y as f32 / 2.0))
}

/// Not an exercise: frame 2 warped back by the flow: pixel `(x, y)` reads `img` at
/// `(x + u, y + v)` (bilinear, edge copied outside).
pub fn warp(img: &Image<f32>, u: &Image<f32>, v: &Image<f32>) -> Image<f32> {
    img.map_xy(|x, y, _| bilinear(img, x as f32 + u[(x, y)], y as f32 + v[(x, y)]))
}
