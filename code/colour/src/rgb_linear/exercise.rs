// Coding exercises for "RGB: gamma, linear maps and white balance". Replace each `todo!()`
// with your code, then check it with `cargo test -p colour rgb_linear::` and look at `cargo run -p colour -- rgb-linear`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{linalg, Image};

/// A colour image: `[red, green, blue]` per pixel, values in [0, 1].
pub type Rgb = Image<[f32; 3]>;

/// C1a. sRGB decoding: a stored value `c` in [0, 1] (the 8-bit code divided by 255) to
/// linear light (slide "The math: the sRGB transfer function").
pub fn srgb_decode(c: f32) -> f32 {
    todo!("C1a: srgb_decode")
}

/// C1b. sRGB encoding: linear light in [0, 1] to the stored value; the inverse of `srgb_decode`.
pub fn srgb_encode(v: f32) -> f32 {
    todo!("C1b: srgb_encode")
}

/// C2. Downsample an encoded colour image by `factor`: every `factor × factor` block becomes
/// one pixel, the average of its pixels. With `linear` false, average the stored values; with
/// `linear` true, decode → average → encode. (The width and height are divided by `factor`;
/// leftover pixels at the right and bottom are dropped.)
///
/// Hint: `Image::from_fn(w / factor, h / factor, |x, y| ...)` and two loops over the block.
pub fn downsample(img: &Rgb, factor: usize, linear: bool) -> Rgb {
    todo!("C2: downsample")
}

/// C3. The RGB → XYZ matrix of a display (`m[row][column]`) from the chromaticities (x, y)
/// of its three primaries and of its white (slide "The math: the matrix from primaries and
/// white point"): primary k has XYZ proportional to `p_k = (x/y, 1, (1 − x − y)/y)`; the
/// scales s solve `[p_R p_G p_B] s = w`, with the white `w` scaled to Y = 1; column k of
/// the matrix is `s_k p_k`.
///
/// Hint: `linalg::solve(&a, &b)` solves the 3 × 3 system (in f64).
pub fn rgb_to_xyz_matrix(primaries: [[f32; 2]; 3], white: [f32; 2]) -> [[f32; 3]; 3] {
    todo!("C3: rgb_to_xyz_matrix")
}

/// C4. Y′CbCr (JPEG) of one pixel with codes 0..255: luma `Y′ = 0.299 R′ + 0.587 G′ + 0.114 B′`,
/// `Cb = 128 + 0.564 (B′ − Y′)`, `Cr = 128 + 0.713 (R′ − Y′)` (not clipped).
pub fn ycbcr(rgb: [f32; 3]) -> [f32; 3] {
    todo!("C4: ycbcr")
}

/// C5. Grey-world white balance of a **linear** colour image: the gains that make the mean
/// of every channel equal to the mean of the green channel, `g_c = mean_G / mean_c`, and the
/// image with every channel multiplied by its gain. Returns `(gains, corrected)`.
pub fn grey_world(img: &Rgb) -> ([f32; 3], Rgb) {
    todo!("C5: grey_world")
}
