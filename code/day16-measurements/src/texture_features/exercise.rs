// Coding exercises for "Texture: co-occurrence matrices". Replace each `todo!()` with your
// code, then check it with `cargo test -p day16 texture_features::` and look at `cargo run -p day16 -- texture-features`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// The five Haralick features of the lecture (slide "Features: statistics of the matrix").
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Haralick {
    pub contrast: f32,
    pub homogeneity: f32,
    pub asm: f32,
    /// In bits (log base 2).
    pub entropy: f32,
    pub correlation: f32,
}

/// C1. Quantise grey values to `levels` levels 0, 1, ..., levels − 1: the range `lo..hi`
/// is cut into `levels` equal parts; values below `lo` go to 0, values from `hi` up to the
/// last level.
///
/// Hint: `(v − lo) / (hi − lo) · levels`, rounded down, then kept between 0 and levels − 1.
pub fn quantise(img: &Image<f32>, levels: u32, lo: f32, hi: f32) -> Image<u32> {
    todo!("C1: quantise")
}

/// C2. The symmetric co-occurrence counts `P` for the offset `(dx, dy)` (`y` down, so 90° is
/// `(0, −d)`): for every pixel whose partner at the offset is also inside the image and
/// inside `mask` (if given), add 1 to `P[i][j]` and 1 to `P[j][i]`, with `i` the level of the
/// pixel and `j` the level of its partner.
///
/// Hint: `vec![vec![0.0; l]; l]` is an `l × l` matrix of zeros; `q.contains(x, y)` takes
/// signed numbers. `mask.is_none_or(|m| m[(x, y)])` is true without a mask.
pub fn glcm(q: &Image<u32>, levels: u32, dx: isize, dy: isize, mask: Option<&Image<bool>>) -> Vec<Vec<f32>> {
    todo!("C2: glcm")
}

/// C3. The Haralick features of a symmetric co-occurrence matrix (counts or probabilities):
/// first divide by the sum to get `p(i, j)`.
///
/// Hint: two passes over the matrix: the first for contrast, homogeneity, ASM, entropy (skip
/// cells with p = 0) and the mean μ; the second for the variance σ² and the covariance.
pub fn haralick(counts: &[Vec<f32>]) -> Haralick {
    todo!("C3: haralick")
}

/// The 8 neighbours of a pixel as offsets `(dx, dy)`, clockwise from the top-left (as the LBP
/// of the lecture reads them).
pub const CLOCKWISE: [(isize, isize); 8] = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)];

/// C4a. The local binary pattern at `(x, y)`: the 8 neighbours clockwise from the top-left,
/// bit 1 if the neighbour is at least the centre, read as a binary number with the first
/// bit most significant. Pixels outside the image count as 0.
///
/// Hint: for each neighbour in `CLOCKWISE` order, `code = (code << 1) | bit`.
pub fn lbp(img: &Image<f32>, x: usize, y: usize) -> u8 {
    todo!("C4a: lbp")
}

/// C4b. The rotation-invariant code: the smallest of the 8 circular rotations of the bits.
///
/// Hint: `code.rotate_left(r)` turns the 8 bits of a `u8` by `r` places.
pub fn rotation_invariant(code: u8) -> u8 {
    todo!("C4b: rotation_invariant")
}
