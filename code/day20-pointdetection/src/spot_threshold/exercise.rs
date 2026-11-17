// Coding exercises for "How many false spots? Thresholds from noise statistics".
// Replace each `todo!()` with your code, then check it with `cargo test -p day20 spot_threshold::`
// and look at `cargo run -p day20 -- spot-threshold`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// The median of a list of numbers (the mean of the two middle ones for an even count).
pub fn median(values: &[f32]) -> f32 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let m = v.len() / 2;
    if v.len() % 2 == 1 { v[m] } else { 0.5 * (v[m - 1] + v[m]) }
}

/// C1. The robust noise estimate: the median absolute deviation from the median, divided by
/// 0.6745 (for normal noise this is the standard deviation).
///
/// Hint: the helper `median` above is given; use it twice.
pub fn mad_sigma(values: &[f32]) -> f32 {
    todo!("C1: mad_sigma")
}

/// C2. The noise sd of an image, estimated from a high-pass version of it: the differences
/// of horizontal neighbours, `d = I(x+1, y) - I(x, y)`. Their sd is √2 times the noise sd
/// (the variances of the two pixels add), and a smooth background drops out.
///
/// Hint: collect the differences in a `Vec<f32>`, then `mad_sigma(&d) / 2f32.sqrt()`.
pub fn estimate_noise(img: &Image<f32>) -> f32 {
    todo!("C2: estimate_noise")
}

/// C3. The local maxima of a response measured in noise sd (`z`) that are above `k`: pixels
/// larger than all 8 neighbours (the outermost rows and columns are skipped). Returns their
/// positions `(x, y)`. With `k = f32::NEG_INFINITY` it returns all local maxima.
///
/// Hint: loop `y` over `1..height - 1` and `x` over `1..width - 1`, so that all
/// 8 neighbours exist.
pub fn maxima_above(z: &Image<f32>, k: f32) -> Vec<(usize, usize)> {
    todo!("C3: maxima_above")
}

/// C4. The expected number of noise maxima above `u` sd in an image of `area` pixels of white
/// noise smoothed by a Gaussian of width `sigma_f` (the Euler-characteristic formula):
/// `area / (2 sigma_f²) · u e^(−u²/2) / (2π)^(3/2)`.
pub fn euler_expected(area: f32, sigma_f: f32, u: f32) -> f32 {
    todo!("C4: euler_expected")
}

/// C5. The Anscombe transform `2 sqrt(x + 3/8)` of every pixel: Poisson counts become values
/// whose noise sd is about 1, whatever the brightness.
///
/// Hint: one line with `img.map`.
pub fn anscombe(img: &Image<f32>) -> Image<f32> {
    todo!("C5: anscombe")
}
