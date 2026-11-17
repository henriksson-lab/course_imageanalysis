// Model solution for the coding exercises of "Segmentation networks".

use imgcourse::Image;

/// C1. The mean per-pixel cross-entropy of a probability map `p` against the true mask `g`:
/// `-ln p` at object pixels, `-ln(1 - p)` at background pixels, averaged over all pixels.
pub fn pixel_cross_entropy(p: &Image<f32>, g: &Image<bool>) -> f32 {
    let mut sum = 0.0;
    for (x, y) in p.coords() {
        sum -= if g[(x, y)] { p[(x, y)].ln() } else { (1.0 - p[(x, y)]).ln() };
    }
    sum / (p.width() * p.height()) as f32
}

/// C2. The soft Dice `2 sum(p g) / (sum(p) + sum(g))` (slide "The math: soft Dice"). For a
/// map of only 0 and 1 it is the ordinary Dice of a mask.
pub fn soft_dice(p: &Image<f32>, g: &Image<bool>) -> f32 {
    let (mut overlap, mut sum_p, mut sum_g) = (0.0, 0.0, 0.0);
    for (x, y) in p.coords() {
        let gv = if g[(x, y)] { 1.0 } else { 0.0 };
        overlap += p[(x, y)] * gv;
        sum_p += p[(x, y)];
        sum_g += gv;
    }
    2.0 * overlap / (sum_p + sum_g)
}

/// C3. The gradient of the soft Dice with respect to every `p_j`:
/// `(2 g_j S - 2 I) / S^2` with `I = sum(p g)` and `S = sum(p) + sum(g)`.
pub fn soft_dice_gradient(p: &Image<f32>, g: &Image<bool>) -> Image<f32> {
    let (mut i, mut s) = (0.0, 0.0);
    for (x, y) in p.coords() {
        let gv = if g[(x, y)] { 1.0 } else { 0.0 };
        i += p[(x, y)] * gv;
        s += p[(x, y)] + gv;
    }
    g.map(|gj| {
        let gv = if gj { 1.0 } else { 0.0 };
        (2.0 * gv * s - 2.0 * i) / (s * s)
    })
}

/// C4. Nearest-neighbour upsampling: every pixel becomes a `factor × factor` block.
pub fn upsample_nearest(img: &Image<f32>, factor: usize) -> Image<f32> {
    Image::from_fn(img.width() * factor, img.height() * factor, |x, y| img[(x / factor, y / factor)])
}

/// C5. Transposed convolution with stride `s` (no padding): every input pixel stamps a copy
/// of the kernel, scaled by its value, into the output with its top-left corner at
/// `(s x, s y)`; overlapping stamps add up. The output is
/// `((w - 1) s + kw) × ((h - 1) s + kh)`.
pub fn transposed_conv(img: &Image<f32>, kernel: &Image<f32>, s: usize) -> Image<f32> {
    let (kw, kh) = (kernel.width(), kernel.height());
    let mut out = Image::new((img.width() - 1) * s + kw, (img.height() - 1) * s + kh, 0.0);
    for (x, y) in img.coords() {
        for v in 0..kh {
            for u in 0..kw {
                out[(s * x + u, s * y + v)] += img[(x, y)] * kernel[(u, v)];
            }
        }
    }
    out
}
