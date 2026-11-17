// Coding exercises for "Ratio imaging and the noise of a ratio". Replace each `todo!()` with
// your code, then check it with `cargo test -p fret ratiometric::` and look at `cargo run -p fret -- ratiometric`.
#![allow(unused_variables, unused_imports)]

use crate::ratiometric::sim::colour_map;
use imgcourse::Image;

/// C1. The ratio image (slide "Background before dividing"): subtract the background from
/// both channels, then R = A / D where the sum of the two subtracted channels is at least
/// `min_sum` (the mask), and NaN ("no value") elsewhere.
pub fn ratio_image(a: &Image<f32>, d: &Image<f32>, bg_a: f32, bg_d: f32, min_sum: f32) -> Image<f32> {
    todo!("C1: ratio_image")
}

/// C2a. The ratio of a region as a ratio of sums: Σ A / Σ D over the mask
/// (slide "Ratios: bias, tails and the right average").
pub fn ratio_of_sums(a: &Image<f32>, d: &Image<f32>, mask: &Image<bool>) -> f32 {
    todo!("C2a: ratio_of_sums")
}

/// C2b. The mean of the pixel ratios A / D over the mask, skipping pixels with D ≤ 0.
pub fn mean_of_ratios(a: &Image<f32>, d: &Image<f32>, mask: &Image<bool>) -> f32 {
    todo!("C2b: mean_of_ratios")
}

/// C3. From ratio to concentration (slide "Calibration: from ratio to concentration"):
/// c = K_d (R − R_min) / (R_max − R).
pub fn calibrate(r: f32, kd: f32, r_min: f32, r_max: f32) -> f32 {
    todo!("C3: calibrate")
}

/// C4. Binning: every `k × k` block of pixels summed into one pixel (the image shrinks by `k`
/// in each direction; pixels that do not fill a whole block at the right or bottom are dropped).
pub fn bin(img: &Image<f32>, k: usize) -> Image<f32> {
    todo!("C4: bin")
}

/// C5. The intensity-modulated display (slide "Displaying a ratio"): the colour from the
/// ratio (`colour_map((R − r_lo) / (r_hi − r_lo))`, given), the lightness from the intensity:
/// the colour times min(intensity / i_max, 1). Pixels without a ratio (NaN) are black.
pub fn intensity_modulated(ratio: &Image<f32>, intensity: &Image<f32>, r_lo: f32, r_hi: f32, i_max: f32) -> Image<[f32; 3]> {
    todo!("C5: intensity_modulated")
}
