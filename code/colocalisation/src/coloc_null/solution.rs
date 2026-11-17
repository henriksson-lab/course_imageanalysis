// Model solution for the coding exercises of "Is it more than chance?".

use crate::common::pearson;
use imgcourse::{Border, Image, Rng};

/// C1. The shift curve (slide "The shift curve"): for every shift dx = −max_shift, …, max_shift,
/// Pearson's r over the mask between A(x, y) and B(x + dx, y). Outside the image, B is read
/// with the nearest edge pixel. Returns the 2 · max_shift + 1 values, the shift 0 in the middle.
pub fn shift_curve(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>, max_shift: usize) -> Vec<f32> {
    let m = max_shift as isize;
    let mut curve = Vec::new();
    for dx in -m..=m {
        let shifted = Image::from_fn(b.width(), b.height(), |x, y| b.get_border(x as isize + dx, y as isize, Border::Clamp));
        curve.push(pearson(a, &shifted, mask));
    }
    curve
}

/// C2. The sub-pixel position of a peak: the vertex of the parabola through the three values
/// at −1, 0 and +1 (slide "The math: the width of the peak"). Returns an offset from 0,
/// between −0.5 and 0.5 when the middle value is the largest.
pub fn subpixel_peak(r_minus: f32, r0: f32, r_plus: f32) -> f32 {
    (r_minus - r_plus) / (2.0 * (r_minus - 2.0 * r0 + r_plus))
}

/// C3. Block randomisation (slide "Randomisation"): cut the image into `block × block`
/// squares (a grid from the top left; squares that do not fit at the right or bottom edge stay
/// where they are), take the squares whose **centre** pixel is in the mask, and put their
/// contents back in a random order (every order equally likely). All other pixels stay.
pub fn block_shuffle(b: &Image<f32>, mask: &Image<bool>, block: usize, rng: &mut Rng) -> Image<f32> {
    let mut corners = Vec::new();
    for by in 0..b.height() / block {
        for bx in 0..b.width() / block {
            let (x0, y0) = (bx * block, by * block);
            if mask[(x0 + block / 2, y0 + block / 2)] {
                corners.push((x0, y0));
            }
        }
    }
    let n = corners.len();
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        order.swap(i, rng.below(i + 1));
    }
    let mut out = b.clone();
    for k in 0..n {
        let (tx, ty) = corners[k]; // the place
        let (sx, sy) = corners[order[k]]; // the square that goes there
        for dy in 0..block {
            for dx in 0..block {
                out[(tx + dx, ty + dy)] = b[(sx + dx, sy + dy)];
            }
        }
    }
    out
}

/// C4. The permutation p-value (slide "The math: the permutation p-value"):
/// (1 + number of null values ≥ r_obs) / (N + 1).
pub fn permutation_p(r_obs: f32, null: &[f32]) -> f32 {
    let at_least = null.iter().filter(|&&r| r >= r_obs).count();
    (1 + at_least) as f32 / (null.len() + 1) as f32
}

/// C5a. The product of the differences from the mean (PDM, slide "Li's ICQ"):
/// (A − mean A)(B − mean B) at every pixel of the mask, with the means over the mask;
/// 0 outside the mask.
pub fn pdm(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>) -> Image<f32> {
    let (mut n, mut sa, mut sb) = (0.0f32, 0.0f32, 0.0f32);
    for p in a.coords().filter(|&p| mask[p]) {
        n += 1.0;
        sa += a[p];
        sb += b[p];
    }
    let (ma, mb) = (sa / n, sb / n);
    a.map_xy(|x, y, va| if mask[(x, y)] { (va - ma) * (b[(x, y)] - mb) } else { 0.0 })
}

/// C5b. Li's ICQ: the share of the mask's pixels with a positive PDM, minus 0.5.
pub fn icq(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>) -> f32 {
    let d = pdm(a, b, mask);
    let (mut n, mut positive) = (0, 0);
    for p in a.coords().filter(|&p| mask[p]) {
        n += 1;
        if d[p] > 0.0 {
            positive += 1;
        }
    }
    positive as f32 / n as f32 - 0.5
}
