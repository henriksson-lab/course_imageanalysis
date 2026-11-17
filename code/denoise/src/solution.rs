// Model solution for the coding exercises of "Denoising, artefacts, and what you may measure".

use imgcourse::{Border, Image};

/// C1. The median filter: every output pixel is the middle value of the
/// `(2 radius + 1) × (2 radius + 1)` window around it (mirror border).
pub fn median_filter(img: &Image<f32>, radius: usize) -> Image<f32> {
    let r = radius as isize;
    let mut out = Image::new(img.width(), img.height(), 0.0);
    let mut window = Vec::new();
    for (x, y) in img.coords() {
        window.clear();
        for dy in -r..=r {
            for dx in -r..=r {
                window.push(img.get_border(x as isize + dx, y as isize + dy, Border::Mirror));
            }
        }
        window.sort_by(f32::total_cmp);
        out[(x, y)] = window[window.len() / 2]; // an odd number of values: the middle one
    }
    out
}

/// C2. The bilateral filter: a weighted mean over the `(2 radius + 1)²` window (mirror border),
/// each neighbour `q` of the pixel `p` weighted by closeness and similarity:
/// `w = exp(−|p − q|² / 2σs²) · exp(−(I(p) − I(q))² / 2σr²)`.
pub fn bilateral(img: &Image<f32>, radius: usize, sigma_s: f32, sigma_r: f32) -> Image<f32> {
    let r = radius as isize;
    let mut out = Image::new(img.width(), img.height(), 0.0);
    for (x, y) in img.coords() {
        let centre = img[(x, y)];
        let (mut sum, mut total) = (0.0, 0.0);
        for dy in -r..=r {
            for dx in -r..=r {
                let v = img.get_border(x as isize + dx, y as isize + dy, Border::Mirror);
                let close = (-((dx * dx + dy * dy) as f32) / (2.0 * sigma_s * sigma_s)).exp();
                let similar = (-(centre - v).powi(2) / (2.0 * sigma_r * sigma_r)).exp();
                sum += close * similar * v;
                total += close * similar;
            }
        }
        out[(x, y)] = sum / total;
    }
    out
}

/// C3. Non-local means: a weighted mean over the pixels `q` of the search window
/// (`(2 search + 1)²` around `p`, only inside the image). The weight compares the patches
/// (`(2 patch + 1)²`, mirror border) around `p` and `q`: `w = exp(−d² / h²)`, with `d²` the
/// mean squared difference of the two patches.
pub fn nonlocal_means(img: &Image<f32>, patch: usize, search: usize, h: f32) -> Image<f32> {
    let (pr, sr) = (patch as isize, search as isize);
    let read = |x: isize, y: isize| img.get_border(x, y, Border::Mirror);
    let patch_size = ((2 * pr + 1) * (2 * pr + 1)) as f32;
    let mut out = Image::new(img.width(), img.height(), 0.0);
    for (x, y) in img.coords() {
        let (px, py) = (x as isize, y as isize);
        let (mut sum, mut total) = (0.0, 0.0);
        for qy in py - sr..=py + sr {
            for qx in px - sr..=px + sr {
                if !img.contains(qx, qy) {
                    continue;
                }
                // how different are the patches around p and q?
                let mut d2 = 0.0;
                for dy in -pr..=pr {
                    for dx in -pr..=pr {
                        d2 += (read(px + dx, py + dy) - read(qx + dx, qy + dy)).powi(2);
                    }
                }
                let w = (-(d2 / patch_size) / (h * h)).exp();
                sum += w * img[(qx as usize, qy as usize)];
                total += w;
            }
        }
        out[(x, y)] = sum / total;
    }
    out
}
