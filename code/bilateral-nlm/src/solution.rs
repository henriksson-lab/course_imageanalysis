// Model solution for the coding exercises of "Bilateral filter and non-local means".

use imgcourse::{Border, Image};

/// C1. A weighted mean and its effective number of averaged values
/// (slide "How many pixels are averaged?"): returns
/// `(sum w v / sum w, (sum w)^2 / sum w^2)`.
pub fn weighted_mean(values: &[f32], weights: &[f32]) -> (f32, f32) {
    let (mut sw, mut swv, mut sww) = (0.0, 0.0, 0.0);
    for (v, w) in values.iter().zip(weights) {
        sw += w;
        swv += w * v;
        sww += w * w;
    }
    (swv / sw, sw * sw / sww)
}

/// C2. The bilateral filter: every output pixel is the weighted mean of the pixels in the
/// `(2 radius + 1)²` window around it, with weight
/// `exp(-distance² / 2σs²) * exp(-(value difference)² / 2σr²)`.
/// Window pixels outside the image are left out.
pub fn bilateral(img: &Image<f32>, sigma_s: f32, sigma_r: f32, radius: usize) -> Image<f32> {
    let r = radius as isize;
    let mut out = Image::new(img.width(), img.height(), 0.0);
    for (x, y) in img.coords() {
        let centre = img[(x, y)];
        let (mut sum, mut total) = (0.0, 0.0);
        for dy in -r..=r {
            for dx in -r..=r {
                let (qx, qy) = (x as isize + dx, y as isize + dy);
                if !img.contains(qx, qy) {
                    continue;
                }
                let v = img[(qx as usize, qy as usize)];
                let close = (-((dx * dx + dy * dy) as f32) / (2.0 * sigma_s * sigma_s)).exp();
                let similar = (-(v - centre).powi(2) / (2.0 * sigma_r * sigma_r)).exp();
                sum += close * similar * v;
                total += close * similar;
            }
        }
        out[(x, y)] = sum / total;
    }
    out
}

/// C3. Non-local means at position `i` of a row (slide "Non-local means by hand"): compare the
/// patch `x[i-half ..= i+half]` with the patch around every other centre `j` whose patch lies
/// inside the row; `d²` = mean squared difference of the two patches, weight `exp(-d² / h²)`;
/// return the weighted mean of the centre values `x[j]`.
pub fn nlm_1d(x: &[f32], i: usize, half: usize, h: f32) -> f32 {
    let m = 2 * half + 1;
    let (mut sum, mut total) = (0.0, 0.0);
    for j in half..x.len() - half {
        let mut d2 = 0.0;
        for k in 0..m {
            let diff = x[i - half + k] - x[j - half + k];
            d2 += diff * diff;
        }
        d2 /= m as f32;
        let w = (-d2 / (h * h)).exp();
        sum += w * x[j];
        total += w;
    }
    sum / total
}

/// C4. Non-local means on an image: for every pixel p, every pixel q in the
/// `(2 search + 1)²` window around p (inside the image) gets the weight `exp(-d² / h²)`, with
/// `d²` the mean squared difference of the `(2 half + 1)²` patches around p and q (patch pixels
/// outside the image read with a mirror border). The output is the weighted mean.
pub fn nlm(img: &Image<f32>, half: usize, search: usize, h: f32) -> Image<f32> {
    let (hp, s) = (half as isize, search as isize);
    let m = ((2 * half + 1) * (2 * half + 1)) as f32;
    let mut out = Image::new(img.width(), img.height(), 0.0);
    for (x, y) in img.coords() {
        let (px, py) = (x as isize, y as isize);
        let (mut sum, mut total) = (0.0, 0.0);
        for qy in py - s..=py + s {
            for qx in px - s..=px + s {
                if !img.contains(qx, qy) {
                    continue;
                }
                let mut d2 = 0.0;
                for ky in -hp..=hp {
                    for kx in -hp..=hp {
                        let a = img.get_border(px + kx, py + ky, Border::Mirror);
                        let b = img.get_border(qx + kx, qy + ky, Border::Mirror);
                        d2 += (a - b) * (a - b);
                    }
                }
                let w = (-(d2 / m) / (h * h)).exp();
                sum += w * img[(qx as usize, qy as usize)];
                total += w;
            }
        }
        out[(x, y)] = sum / total;
    }
    out
}
