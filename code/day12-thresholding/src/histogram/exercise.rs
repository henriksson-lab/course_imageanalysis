// Coding exercises for "Histograms and a global threshold". Replace each `todo!()` with your
// code, then check it with `cargo test -p day12 histogram::` and look at `cargo run -p day12 -- histogram`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// The bin of a grey value `v` in a histogram with `bins` equal bins over [0, 1] (given, in
/// `src/common.rs`): bin `i` holds the values from `i / bins` up to (not including)
/// `(i + 1) / bins`. The value 1 goes into the last bin, values outside [0, 1] into the first
/// or last bin. With 256 bins, an 8-bit grey level `k` (stored as `k / 255`) lands in bin `k`.
pub use crate::common::bin_of;

/// C1. The histogram: `h[i]` = the number of pixels whose grey value falls into bin `i`.
///
/// Hint: start with `vec![0; bins]` and add 1 to `h[bin_of(img[(x, y)], bins)]` for every pixel.
pub fn histogram(img: &Image<f32>, bins: usize) -> Vec<usize> {
    todo!("C1: histogram")
}

/// C2. The cumulative histogram, normalised: `p[t]` = the fraction of pixels in bins `0..=t`
/// (the slide "The cumulative histogram and the percentile threshold"). The last value is 1.
///
/// Hint: keep a running sum of the counts; divide by the total `h.iter().sum::<usize>()`.
pub fn cumulative(h: &[usize]) -> Vec<f32> {
    todo!("C2: cumulative")
}

/// C3. Threshold at bin `t`: object (true) where the pixel's bin is above `t`, background
/// (false) where it is `t` or below.
///
/// Hint: one line with `img.map(...)`.
pub fn threshold(img: &Image<f32>, bins: usize, t: usize) -> Image<bool> {
    todo!("C3: threshold")
}

/// C4. The percentile threshold: the smallest `t` that leaves at most the fraction `q` of the
/// pixels above `t` ("keep the brightest q").
///
/// Hint: count in pixels, not in fractions: start with all `N` pixels above, and subtract
/// `h[t]` for t = 0, 1, 2, ... until at most `q · N` are left.
pub fn percentile_threshold(h: &[usize], q: f32) -> usize {
    todo!("C4: percentile_threshold")
}

/// C5. The mean and the variance of the grey level of a random pixel, in bin numbers
/// (slide "The math: the histogram as a distribution"): mean = sum of i p_i,
/// variance = sum of (i - mean)² p_i, with p_i = h_i / N.
pub fn mean_variance(h: &[usize]) -> (f32, f32) {
    todo!("C5: mean_variance")
}
