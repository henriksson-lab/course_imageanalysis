// Model solution for the coding exercises of "Scale space, blobs and pyramids".

use crate::gaussian_blur;
use imgcourse::{Border, Image};

/// C1a. The scale-normalised Laplacian of Gaussian, sign flipped so that bright blobs are
/// positive: −σ² (Lxx + Lyy), where L is the image blurred at `sigma` and the second
/// derivatives are the differences [1 −2 1] along x and along y (copy the edge outside).
pub fn normalised_log(img: &Image<f32>, sigma: f32) -> Image<f32> {
    let l = gaussian_blur(img, sigma);
    l.map_xy(|x, y, c| {
        let at = |dx: isize, dy: isize| l.get_border(x as isize + dx, y as isize + dy, Border::Clamp);
        let lxx = at(-1, 0) - 2.0 * c + at(1, 0);
        let lyy = at(0, -1) - 2.0 * c + at(0, 1);
        -sigma * sigma * (lxx + lyy)
    })
}

/// C1b. The zero crossings of a row of values (Marr–Hildreth): every place where the sign
/// changes between element `i` and `i + 1`, as a position between them by linear
/// interpolation, `i + a / (a − b)` for the values `a` and `b`.
pub fn zero_crossings(row: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    for i in 0..row.len().saturating_sub(1) {
        let (a, b) = (row[i], row[i + 1]);
        if (a > 0.0 && b < 0.0) || (a < 0.0 && b > 0.0) {
            out.push(i as f32 + a / (a - b));
        }
    }
    out
}

/// One detected blob: its centre, its scale, and the normalised response there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Blob {
    pub x: usize,
    pub y: usize,
    pub sigma: f32,
    pub response: f32,
}

/// C2. Blob detection: `stack[k]` is the normalised LoG at scale `sigmas[k]` (all the same
/// size). A blob is a pixel whose response is above `threshold` and larger than all its
/// neighbours in (x, y, σ): 8 in its own scale, 9 in the scale below and 9 above (only 9 + 8
/// at the smallest and the largest scale). Pixels on the edge of the image are skipped: they
/// do not have all their neighbours, and the copied edge makes their second differences wrong.
pub fn detect_blobs(stack: &[Image<f32>], sigmas: &[f32], threshold: f32) -> Vec<Blob> {
    let mut blobs = Vec::new();
    for k in 0..stack.len() {
        for (x, y) in stack[k].coords() {
            let (w, h) = (stack[k].width(), stack[k].height());
            let v = stack[k][(x, y)];
            if x == 0 || y == 0 || x == w - 1 || y == h - 1 || v <= threshold {
                continue;
            }
            let mut is_max = true;
            for dk in -1..=1isize {
                let kk = k as isize + dk;
                if kk < 0 || kk >= stack.len() as isize {
                    continue;
                }
                for dy in -1..=1isize {
                    for dx in -1..=1isize {
                        let (nx, ny) = ((x as isize + dx) as usize, (y as isize + dy) as usize);
                        let me = dk == 0 && dx == 0 && dy == 0;
                        if !me && stack[kk as usize][(nx, ny)] >= v {
                            is_max = false;
                        }
                    }
                }
            }
            if is_max {
                blobs.push(Blob { x, y, sigma: sigmas[k], response: v });
            }
        }
    }
    blobs
}

/// C3. The difference of Gaussians: the image blurred at `k·sigma` minus the image blurred
/// at `sigma`. It is close to (k − 1) σ² ∇²L, i.e. to −(k − 1) times `normalised_log`.
pub fn dog(img: &Image<f32>, sigma: f32, k: f32) -> Image<f32> {
    let wide = gaussian_blur(img, k * sigma);
    let narrow = gaussian_blur(img, sigma);
    wide.zip_map(&narrow, |a, b| a - b)
}

/// C4a. One level down the Gaussian pyramid: blur with σ = 1, then keep every second pixel
/// in each direction (pixels (0, 0), (2, 0), …). A 2w × 2h image becomes w × h.
pub fn downsample(img: &Image<f32>) -> Image<f32> {
    let b = gaussian_blur(img, 1.0);
    Image::from_fn(img.width().div_ceil(2), img.height().div_ceil(2), |x, y| b[(2 * x, 2 * y)])
}

/// C4b. Back up to a `width × height` image: pixel (x, y) reads the small image at
/// (x/2, y/2) by bilinear interpolation (copy the edge outside).
pub fn upsample(small: &Image<f32>, width: usize, height: usize) -> Image<f32> {
    Image::from_fn(width, height, |x, y| {
        let (sx, sy) = (x as f32 / 2.0, y as f32 / 2.0);
        let (x0, y0) = (sx.floor() as isize, sy.floor() as isize);
        let (a, b) = (sx - x0 as f32, sy - y0 as f32);
        let p = |dx: isize, dy: isize| small.get_border(x0 + dx, y0 + dy, Border::Clamp);
        (1.0 - a) * (1.0 - b) * p(0, 0) + a * (1.0 - b) * p(1, 0) + (1.0 - a) * b * p(0, 1) + a * b * p(1, 1)
    })
}

/// C4c. The Laplacian pyramid with `levels` levels: level i is g_i − up(g_(i+1)), where g_0 is
/// the image and g_(i+1) = downsample(g_i); the last level is the small Gaussian level itself.
pub fn laplacian_pyramid(img: &Image<f32>, levels: usize) -> Vec<Image<f32>> {
    let mut out = Vec::new();
    let mut g = img.clone();
    for _ in 0..levels - 1 {
        let next = downsample(&g);
        let up = upsample(&next, g.width(), g.height());
        out.push(g.zip_map(&up, |a, b| a - b));
        g = next;
    }
    out.push(g);
    out
}

/// C4d. Rebuild the image from its Laplacian pyramid: start with the last level, and going
/// back up, upsample and add the level: g_i = l_i + up(g_(i+1)).
pub fn reconstruct(pyramid: &[Image<f32>]) -> Image<f32> {
    let mut g = pyramid[pyramid.len() - 1].clone();
    for l in pyramid[..pyramid.len() - 1].iter().rev() {
        let up = upsample(&g, l.width(), l.height());
        g = l.zip_map(&up, |a, b| a + b);
    }
    g
}
