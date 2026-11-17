// Coding exercises for "Lucas–Kanade". Replace each `todo!()` with your code,
// then check it with `cargo test -p day25 lucas_kanade::` and look at `cargo run -p day25 -- lucas-kanade`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};

/// C1. The derivatives of a frame pair: `Ix`, `Iy` by central differences of the **mean** of
/// the two frames (copy the edge outside), and `It = f2 − f1`.
///
/// Hint: `f1.zip_map(f2, |a, b| 0.5 * (a + b))` is the mean; read neighbours with
/// `get_border(.., Border::Clamp)`.
pub fn gradients(f1: &Image<f32>, f2: &Image<f32>) -> (Image<f32>, Image<f32>, Image<f32>) {
    todo!("C1: gradients")
}

/// C2a. Solve the normal equations `J (u, v) = −b` of one window, with `J` given as
/// `[Σ Ix², Σ Ix Iy, Σ Iy²]` and `b = [Σ Ix It, Σ Iy It]`. `None` if `|det J| < 1e-12`.
///
/// Hint: the inverse of `[[a, c], [c, d]]` is `[[d, −c], [−c, a]] / det`; the flow is `−J⁻¹ b`.
pub fn lk_solve(j: [f32; 3], b: [f32; 2]) -> Option<(f32, f32)> {
    todo!("C2a: lk_solve")
}

/// C2b. Lucas–Kanade for one window given as a list of pixels `[Ix, Iy, It]`: add up `J`
/// and `b`, then `lk_solve`.
pub fn lk_window(pixels: &[[f32; 3]]) -> Option<(f32, f32)> {
    todo!("C2b: lk_window")
}

/// C3. The smaller eigenvalue μ2 of `J = [[a, c], [c, d]]`, given as `[a, c, d]`: the confidence.
///
/// Hint: μ2 = (a + d)/2 − sqrt(((a − d)/2)² + c²).
pub fn min_eigenvalue(j: [f32; 3]) -> f32 {
    todo!("C3: min_eigenvalue")
}

/// C4. Lucas–Kanade at every pixel, with a Gaussian window of width `rho`: the five sums of
/// `J` and `b` are five Gaussian blurs of products (`gaussian_blur`), then one solve per pixel.
/// Returns `(u, v, mu2)`; where the solve fails, the flow is 0.
///
/// Hint: `gaussian_blur(&ix.zip_map(&it, |a, b| a * b), rho)` is the sum Σ w Ix It at every pixel.
pub fn lucas_kanade(f1: &Image<f32>, f2: &Image<f32>, rho: f32) -> (Image<f32>, Image<f32>, Image<f32>) {
    todo!("C4: lucas_kanade")
}

/// C5. Coarse to fine with warping: build pyramids of both frames with `levels` levels
/// (`downsample`), start with zero flow at the coarsest level; on every level (coarsest
/// first), `iterations` times: warp frame 2 back by the current flow (`warp`), estimate the
/// remaining flow with `lucas_kanade`, add it. Going one level finer: `upsample_flow`.
///
/// Hint: keep the pyramids in two `Vec`s, level 0 = full size. Loop `for l in (0..levels).rev()`.
pub fn pyramid_lk(f1: &Image<f32>, f2: &Image<f32>, levels: usize, iterations: usize, rho: f32) -> (Image<f32>, Image<f32>) {
    todo!("C5: pyramid_lk")
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

/// Not an exercise: bilinear interpolation, the edge copied outside the image.
pub fn bilinear(img: &Image<f32>, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let p = |i: f32, j: f32| img.get_border(i as isize, j as isize, Border::Clamp);
    (1.0 - fy) * ((1.0 - fx) * p(x0, y0) + fx * p(x0 + 1.0, y0)) + fy * ((1.0 - fx) * p(x0, y0 + 1.0) + fx * p(x0 + 1.0, y0 + 1.0))
}
