// Coding exercises for "Moments as projections onto monomials".
// Replace each `todo!()` with your code, then check it with `cargo test -p day18 moments_basis::`
// and look at `cargo run -p day18 -- moments-basis`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// A table of moments: `m[p][q]` is the moment of order `p + q` with the weight `x^p y^q`.
/// Only the entries with `p + q <= order` are filled; the others stay 0.
pub type Moments = Vec<Vec<f64>>;

/// C1. The raw moments `m[p][q] = sum over all pixels of x^p y^q I(x, y)` for `p + q <= order`,
/// with `x` the column and `y` the row in pixel units (slide "One weight per moment").
///
/// Hint: start with `vec![vec![0.0; order + 1]; order + 1]`; for every pixel add `x^p y^q I` to
/// `m[p][q]` for all `p + q <= order` (`(x as f64).powi(p as i32)`).
pub fn raw_moments(img: &Image<f32>, order: usize) -> Moments {
    todo!("C1: raw_moments")
}

/// C2. The central moments `mu[p][q] = sum of (x - xbar)^p (y - ybar)^q I(x, y)`, with the
/// centroid `xbar = m10 / m00`, `ybar = m01 / m00`: a first pass for the centroid, a second
/// pass for the sums.
///
/// Hint: get the centroid from `raw_moments(img, 1)`, then loop over the pixels again with `x -
/// xbar` and `y - ybar`.
pub fn central_moments(img: &Image<f32>, order: usize) -> Moments {
    todo!("C2: central_moments")
}

/// C3. The image as a distribution: skewness and kurtosis of the two marginals, from a table
/// of central moments up to order 4. Returns `(skew_x, kurt_x, skew_y, kurt_y)` with
/// `skew_x = mu30 / (m00 sd_x^3)`, `kurt_x = mu40 / (m00 sd_x^4)`, `sd_x^2 = mu20 / m00`.
///
/// Hint: `sd_x = (mu[2][0] / mu[0][0]).sqrt()`; the y versions use `mu[0][2]`, `mu[0][3]`,
/// `mu[0][4]`.
pub fn skewness_kurtosis(mu: &Moments) -> (f64, f64, f64, f64) {
    todo!("C3: skewness_kurtosis")
}

/// C4. All raw moments at once, separably (slide "Computing moments"): first the column sums
/// weighted by `y^q`, `c[x][q] = sum over y of y^q I(x, y)`, then `m[p][q] = sum over x of
/// x^p c[x][q]`. With `scaled`, the pixel centres are mapped to [-1, 1] first:
/// `x -> (2x + 1) / width - 1`, `y -> (2y + 1) / height - 1`.
///
/// Hint: `c` is a `Vec` of `width` rows of `order + 1` numbers. Write a small function
/// (closure) that turns a pixel index into a coordinate, so that the same loops work for both
/// choices.
pub fn moments_separable(img: &Image<f32>, order: usize, scaled: bool) -> Moments {
    todo!("C4: moments_separable")
}
