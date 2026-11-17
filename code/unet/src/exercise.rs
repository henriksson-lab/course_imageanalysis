// Coding exercises for "Segmentation networks". Replace each `todo!()` with your code,
// then check it with `cargo test -p unet` and look at `cargo run -p unet`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// C1. The mean per-pixel cross-entropy of a probability map `p` against the true mask `g`:
/// `-ln p` at object pixels, `-ln(1 - p)` at background pixels, averaged over all pixels.
///
/// Hint: loop over `p.coords()`; `g[(x, y)]` is `true` at object pixels; `.ln()` is the
/// natural logarithm.
pub fn pixel_cross_entropy(p: &Image<f32>, g: &Image<bool>) -> f32 {
    todo!("C1: pixel_cross_entropy")
}

/// C2. The soft Dice `2 sum(p g) / (sum(p) + sum(g))` (slide "The math: soft Dice"). For a
/// map of only 0 and 1 it is the ordinary Dice of a mask.
///
/// Hint: three sums in one loop: `p g`, `p` and `g` (count `true` as 1, `false` as 0).
pub fn soft_dice(p: &Image<f32>, g: &Image<bool>) -> f32 {
    todo!("C2: soft_dice")
}

/// C3. The gradient of the soft Dice with respect to every `p_j`:
/// `(2 g_j S - 2 I) / S^2` with `I = sum(p g)` and `S = sum(p) + sum(g)`.
///
/// Hint: first compute `I` and `S` in one loop, then `g.map(..)` gives an image of the same
/// size with one value per pixel.
pub fn soft_dice_gradient(p: &Image<f32>, g: &Image<bool>) -> Image<f32> {
    todo!("C3: soft_dice_gradient")
}

/// C4. Nearest-neighbour upsampling: every pixel becomes a `factor × factor` block.
///
/// Hint: output pixel `(x, y)` comes from input pixel `(x / factor, y / factor)`
/// (whole-number division).
pub fn upsample_nearest(img: &Image<f32>, factor: usize) -> Image<f32> {
    todo!("C4: upsample_nearest")
}

/// C5. Transposed convolution with stride `s` (no padding): every input pixel stamps a copy
/// of the kernel, scaled by its value, into the output with its top-left corner at
/// `(s x, s y)`; overlapping stamps add up. The output is
/// `((w - 1) s + kw) × ((h - 1) s + kh)`.
///
/// Hint: start from an output of zeros; for every input pixel `(x, y)` and every kernel
/// position `(u, v)`, add `img[(x, y)] * kernel[(u, v)]` to `out[(s x + u, s y + v)]`.
pub fn transposed_conv(img: &Image<f32>, kernel: &Image<f32>, s: usize) -> Image<f32> {
    todo!("C5: transposed_conv")
}
