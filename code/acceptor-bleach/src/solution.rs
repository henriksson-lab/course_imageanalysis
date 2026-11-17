// Model solution for the coding exercises of "Acceptor photobleaching".

use imgcourse::{Border, Image};

/// C1a. The efficiency at every pixel (slide "How much brighter?"): E = 1 − pre / post;
/// 0 where `post` is not positive.
pub fn efficiency_map(pre: &Image<f32>, post: &Image<f32>) -> Image<f32> {
    pre.zip_map(post, |p, q| if q > 0.0 { 1.0 - p / q } else { 0.0 })
}

/// C1b. The efficiency of a region as a ratio of sums (slide "Pixel-wise or region-wise?"):
/// E = 1 − Σ pre / Σ post over the mask.
pub fn region_efficiency(pre: &Image<f32>, post: &Image<f32>, mask: &Image<bool>) -> f32 {
    let (mut sp, mut sq) = (0.0f64, 0.0f64);
    for p in pre.coords().filter(|&p| mask[p]) {
        sp += pre[p] as f64;
        sq += post[p] as f64;
    }
    (1.0 - sp / sq) as f32
}

/// C2a. The correction for incomplete bleaching (slide "Incomplete bleaching"): with a
/// fraction `b` of the acceptor bleached, E = E_obs / (b + (1 − b) E_obs).
pub fn correct_incomplete(e_obs: f32, b: f32) -> f32 {
    e_obs / (b + (1.0 - b) * e_obs)
}

/// C2b. The control ratio for donor bleaching (slide "Donor bleaching, motion,
/// photoconversion"): Σ post / Σ pre over a region that was not bleached. Dividing the post
/// image by it undoes the donor's own bleaching.
pub fn control_ratio(pre: &Image<f32>, post: &Image<f32>, control: &Image<bool>) -> f32 {
    let (mut sp, mut sq) = (0.0f64, 0.0f64);
    for p in pre.coords().filter(|&p| control[p]) {
        sp += pre[p] as f64;
        sq += post[p] as f64;
    }
    (sq / sp) as f32
}

/// C3a. A whole-pixel shift: the result at (x, y) is `img` at (x + dx, y + dy), 0 outside.
pub fn shift(img: &Image<f32>, dx: isize, dy: isize) -> Image<f32> {
    img.map_xy(|x, y, _| img.get_border(x as isize + dx, y as isize + dy, Border::Zero))
}

/// C3b. How far did the cells move between `before` and `after`? The whole-pixel shift
/// (dx, dy), each from −max_shift to max_shift, for which `shift(after, dx, dy)` best matches
/// `before`: the largest Σ before(x, y) · after(x + dx, y + dy). Then
/// `shift(after, dx, dy)` is aligned with `before`.
pub fn find_shift(before: &Image<f32>, after: &Image<f32>, max_shift: isize) -> (isize, isize) {
    let mut best = (f64::NEG_INFINITY, 0, 0);
    for dy in -max_shift..=max_shift {
        for dx in -max_shift..=max_shift {
            let mut score = 0.0f64;
            for (x, y) in before.coords() {
                score += (before[(x, y)] * after.get_border(x as isize + dx, y as isize + dy, Border::Zero)) as f64;
            }
            if score > best.0 {
                best = (score, dx, dy);
            }
        }
    }
    (best.1, best.2)
}

/// C4. The noise of E = 1 − P/Q for photon counts P (before) and Q (after)
/// (slide "The math: the noise of E = 1 − P/Q"): σ_E ≈ (P/Q) sqrt(1/P + 1/Q).
pub fn sigma_e(p: f32, q: f32) -> f32 {
    p / q * (1.0 / p + 1.0 / q).sqrt()
}
