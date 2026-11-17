// Coding exercises for "HSV, CIELAB and colour segmentation". Replace each `todo!()` with
// your code, then check it with `cargo test -p day42 colour_perceptual::` and look at
// `cargo run -p day42 -- colour-perceptual`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// A colour image: `[red, green, blue]` per pixel, values in [0, 1].
pub type Rgb = Image<[f32; 3]>;

/// sRGB decoding (from the last unit): stored value in [0, 1] → linear light. In `src/common.rs`.
pub use crate::common::srgb_decode;

/// C1. Hue (degrees in [0, 360)), saturation and value of a colour with R, G, B in [0, 1]
/// (slide "The math: computing H, S, V"). For a grey (max = min) return hue 0 and
/// saturation 0; for black also saturation 0.
///
/// Hint: `(x).rem_euclid(6.0)` is "x mod 6" that is never negative.
pub fn rgb_to_hsv(c: [f32; 3]) -> [f32; 3] {
    todo!("C1: rgb_to_hsv")
}

/// C2. The circular mean of hues in degrees: the angle of the mean of the unit vectors
/// (cos H, sin H), in [0, 360), and the length R of that mean vector (1: all hues agree,
/// near 0: spread out). Returns `(mean, r)`.
///
/// Hint: `f32::atan2(y, x)` gives the angle of (x, y) in radians, between −π and π.
pub fn circular_mean(hues: &[f32]) -> (f32, f32) {
    todo!("C2: circular_mean")
}

/// C3a. CIELAB (L*, a*, b*) of an encoded sRGB colour (values in [0, 1]), with the D65 white
/// (0.9505, 1, 1.089) (slide "The math: from XYZ to Lab"): decode, multiply by the sRGB
/// matrix to get XYZ, then
/// `L* = 116 f(Y/Yn) − 16`, `a* = 500 (f(X/Xn) − f(Y/Yn))`, `b* = 200 (f(Y/Yn) − f(Z/Zn))`.
///
/// Hint: the sRGB matrix rows are (0.4124, 0.3576, 0.1805), (0.2126, 0.7152, 0.0722),
/// (0.0193, 0.1192, 0.9505). f(t) = t^(1/3) above (6/29)³, else t / (3 (6/29)²) + 4/29.
pub fn srgb_to_lab(c: [f32; 3]) -> [f32; 3] {
    todo!("C3a: srgb_to_lab")
}

/// C3b. ΔE*ab: the Euclidean distance between two Lab colours.
pub fn delta_e(lab1: [f32; 3], lab2: [f32; 3]) -> f32 {
    todo!("C3b: delta_e")
}

/// C4. A hue wedge: true where the hue lies in the interval from `h_lo` to `h_hi` degrees
/// (going up from `h_lo`; if `h_lo > h_hi` the interval wraps through 0°, as 340°–20°), and
/// the saturation and value are at least `s_min` and `v_min`.
pub fn hue_mask(img: &Rgb, h_lo: f32, h_hi: f32, s_min: f32, v_min: f32) -> Image<bool> {
    todo!("C4: hue_mask")
}

/// C5. k-means of 3D points: starting from the given centres, repeat `iterations` times:
/// give every point to its nearest centre (squared Euclidean distance), then move each centre
/// to the mean of its points (a centre without points stays where it is).
/// Returns `(centres, labels)`, with `labels[i]` the centre of point i.
pub fn kmeans(points: &[[f32; 3]], centres: &[[f32; 3]], iterations: usize) -> (Vec<[f32; 3]>, Vec<usize>) {
    todo!("C5: kmeans")
}
