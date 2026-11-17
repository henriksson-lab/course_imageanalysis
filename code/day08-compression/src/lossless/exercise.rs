// Coding exercises for "Prediction, dictionaries and lossless formats". Replace each `todo!()`
// with your code, then check it with `cargo test -p day08 lossless::` and look at `cargo run -p day08 -- lossless`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};

/// The predictors of the lecture (slide "Predictors"), from the neighbours a = left, b = up,
/// c = up-left.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Predictor {
    /// No prediction: the residual is the pixel itself.
    None,
    Left,
    Up,
    /// floor((a + b) / 2)
    Average,
    /// a + b − c
    Planar,
    /// Of a, b, c the one closest to a + b − c (ties: a, then b). PNG.
    Paeth,
    /// The median edge detector of JPEG-LS.
    Med,
}

pub const ALL: [Predictor; 7] =
    [Predictor::None, Predictor::Left, Predictor::Up, Predictor::Average, Predictor::Planar, Predictor::Paeth, Predictor::Med];

/// C1. The prediction of a pixel from its neighbours a (left), b (up) and c (up-left).
///
/// Hint: `match predictor { Predictor::None => 0, Predictor::Left => a, ... }`. For the
/// average, `(a + b).div_euclid(2)` rounds down also for negative sums.
pub fn predict(a: i32, b: i32, c: i32, predictor: Predictor) -> i32 {
    todo!("C1: predict")
}

/// C2. The residual image e = x − x̂, scanning row by row. Neighbours outside the image count
/// as 0, so the first pixel is stored as it is (its prediction is 0).
///
/// Hint: `img.get_border(x - 1, y, Border::Zero)` (with `x`, `y` as `isize`) reads the left
/// neighbour, 0 outside.
pub fn residuals(img: &Image<i32>, predictor: Predictor) -> Image<i32> {
    todo!("C2: residuals")
}

/// C3. Decoding: rebuild the image from its residuals. Scan in the same order as the encoder,
/// predict from the pixels already decoded, and add the residual back.
///
/// Hint: read the neighbours from the output image you are filling in, not from `res`.
pub fn decode(res: &Image<i32>, predictor: Predictor) -> Image<i32> {
    todo!("C3: decode")
}

/// C4. The entropy in bits per value of a list of whole numbers (any range, also negative):
/// count how often each value occurs, then H = −Σ p log2 p.
///
/// Hint: find the smallest and largest value; then value `v` can be counted in `counts[v - lo]`.
pub fn entropy(values: &[i32]) -> f32 {
    todo!("C4: entropy")
}
