// Model solution for the coding exercises of "Finding spots: filter, then local maxima".

use crate::spot_filters::sim::mad_sigma;
use imgcourse::{Border, Image};

/// C1. The local maxima of `img` in windows of radius `r` (a (2r+1) × (2r+1) square): the
/// pixels that are at least as large as every pixel in their window. Pixels outside the image
/// are ignored. Tie rule: of equal values, only the first in raster order (row by row) counts,
/// so a pixel must be **larger** than the pixels before it and **at least as large** as the
/// pixels after it. Returns the positions `(x, y)` in raster order.
pub fn local_maxima(img: &Image<f32>, r: usize) -> Vec<(usize, usize)> {
    let r = r as isize;
    let mut out = Vec::new();
    for (x, y) in img.coords() {
        let v = img[(x, y)];
        let mut is_max = true;
        for dy in -r..=r {
            for dx in -r..=r {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if (dx, dy) == (0, 0) || !img.contains(nx, ny) {
                    continue;
                }
                let w = img[(nx as usize, ny as usize)];
                let earlier = dy < 0 || (dy == 0 && dx < 0); // before (x, y) in raster order
                if w > v || (earlier && w == v) {
                    is_max = false;
                }
            }
        }
        if is_max {
            out.push((x, y));
        }
    }
    out
}

/// C2. The spot template as a filter kernel: a Gaussian of width `sigma` on a square of radius
/// `r = ceil(3 sigma)` (size 2r+1), scaled so that its weights sum to 1. With `zero_mean`, its
/// mean is then subtracted from every weight, so that the weights sum to 0 (centre–surround).
/// The centre of the kernel is pixel `(r, r)`.
pub fn spot_kernel(sigma: f32, zero_mean: bool) -> Image<f32> {
    let r = (3.0 * sigma).ceil() as usize;
    let size = 2 * r + 1;
    let mut k = Image::from_fn(size, size, |x, y| {
        let (dx, dy) = (x as f32 - r as f32, y as f32 - r as f32);
        (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp()
    });
    let sum = k.sum();
    k = k.map(|w| w / sum); // the matched filter, normalised: weights sum to 1
    if zero_mean {
        let mean = k.mean();
        k = k.map(|w| w - mean); // the projection onto the zero-sum filters
    }
    k
}

/// C3. Filter `img` with a kernel of odd size, centred on its middle pixel, without flipping
/// it (correlation, as in the kernels unit). Outside the image: mirror.
pub fn correlate(img: &Image<f32>, kernel: &Image<f32>) -> Image<f32> {
    let (rx, ry) = ((kernel.width() / 2) as isize, (kernel.height() / 2) as isize);
    img.map_xy(|x, y, _| {
        let mut sum = 0.0;
        for (kx, ky) in kernel.coords() {
            let (dx, dy) = (kx as isize - rx, ky as isize - ry);
            sum += kernel[(kx, ky)] * img.get_border(x as isize + dx, y as isize + dy, Border::Mirror);
        }
        sum
    })
}

/// C4. The à trous decomposition with `levels` levels: `a0 = img`, `a_j` = `a_(j-1)` blurred
/// along rows and then columns with the taps (1, 4, 6, 4, 1)/16 spaced `2^(j-1)` pixels apart
/// (mirror outside), `w_j = a_(j-1) - a_j`. Returns `([w1, w2, ...], a_levels)`.
pub fn atrous(img: &Image<f32>, levels: usize) -> (Vec<Image<f32>>, Image<f32>) {
    const H: [f32; 5] = [1. / 16., 4. / 16., 6. / 16., 4. / 16., 1. / 16.];
    let mut a = img.clone();
    let mut ws = Vec::new();
    for j in 1..=levels {
        let step = 1isize << (j - 1); // the "holes": taps 2^(j-1) apart
        let pass = |src: &Image<f32>, along_x: bool| {
            src.map_xy(|x, y, _| {
                let mut sum = 0.0;
                for (i, h) in H.iter().enumerate() {
                    let d = (i as isize - 2) * step;
                    let (xx, yy) = if along_x { (x as isize + d, y as isize) } else { (x as isize, y as isize + d) };
                    sum += h * src.get_border(xx, yy, Border::Mirror);
                }
                sum
            })
        };
        let next = pass(&pass(&a, true), false);
        ws.push(a.zip_map(&next, |p, q| p - q));
        a = next;
    }
    (ws, a)
}

/// C5. The multiscale product of the levels `use_levels` (counted from 1, as w1, w2, ...): each
/// level is kept only where it exceeds `k` times its own noise (`mad_sigma` of its values) and
/// set to 0 elsewhere; the kept levels are multiplied pixel by pixel.
pub fn multiscale_product(ws: &[Image<f32>], use_levels: &[usize], k: f32) -> Image<f32> {
    let mut p = Image::new(ws[0].width(), ws[0].height(), 1.0);
    for &j in use_levels {
        let w = &ws[j - 1];
        let t = k * mad_sigma(w.pixels());
        p = p.zip_map(w, |acc, v| if v > t { acc * v } else { 0.0 });
    }
    p
}
