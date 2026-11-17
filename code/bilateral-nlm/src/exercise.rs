// Coding exercises for "Bilateral filter and non-local means". Replace each `todo!()` with your
// code, then check it with `cargo test -p bilateral-nlm` and look at `cargo run -p bilateral-nlm`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};

/// C1. A weighted mean and its effective number of averaged values
/// (slide "How many pixels are averaged?"): returns
/// `(sum w v / sum w, (sum w)^2 / sum w^2)`.
pub fn weighted_mean(values: &[f32], weights: &[f32]) -> (f32, f32) {
    todo!("C1: weighted_mean")
}

/// C2. The bilateral filter: every output pixel is the weighted mean of the pixels in the
/// `(2 radius + 1)²` window around it, with weight
/// `exp(-distance² / 2σs²) * exp(-(value difference)² / 2σr²)`.
/// Window pixels outside the image are left out.
///
/// Hint: two loops over `dy` and `dx` from `-radius` to `radius`; skip the offsets where
/// `img.contains(x + dx, y + dy)` is false (as `isize`). Add up `w * value` and `w`, divide at the end.
pub fn bilateral(img: &Image<f32>, sigma_s: f32, sigma_r: f32, radius: usize) -> Image<f32> {
    todo!("C2: bilateral")
}

/// C3. Non-local means at position `i` of a row (slide "Non-local means by hand"): compare the
/// patch `x[i-half ..= i+half]` with the patch around every other centre `j` whose patch lies
/// inside the row; `d²` = mean squared difference of the two patches, weight `exp(-d² / h²)`;
/// return the weighted mean of the centre values `x[j]`.
///
/// Hint: the candidate centres are `half .. x.len() - half`; the own centre `i` is one of them
/// (with `d² = 0`, weight 1).
pub fn nlm_1d(x: &[f32], i: usize, half: usize, h: f32) -> f32 {
    todo!("C3: nlm_1d")
}

/// C4. Non-local means on an image: for every pixel p, every pixel q in the
/// `(2 search + 1)²` window around p (inside the image) gets the weight `exp(-d² / h²)`, with
/// `d²` the mean squared difference of the `(2 half + 1)²` patches around p and q (patch pixels
/// outside the image read with a mirror border). The output is the weighted mean.
///
/// Hint: four nested loops: the search window (`qy`, `qx`), and inside it the patch offsets
/// (`ky`, `kx`) to add up `d²`. Read patch pixels with `img.get_border(.., Border::Mirror)`.
pub fn nlm(img: &Image<f32>, half: usize, search: usize, h: f32) -> Image<f32> {
    todo!("C4: nlm")
}
