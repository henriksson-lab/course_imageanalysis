// Coding exercises for "Colour for display: colour maps, merges and decorrelation". Replace
// each `todo!()` with your code, then check it with `cargo test -p colour-display` and look at
// `cargo run -p colour-display`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{linalg, Image};

/// A colour image: `[red, green, blue]` per pixel, encoded sRGB values in [0, 1].
pub type Rgb = Image<[f32; 3]>;

/// Viridis, 9 equally spaced anchor colours (encoded sRGB), from t = 0 to t = 1.
pub const VIRIDIS: [[f32; 3]; 9] = [
    [0.267, 0.004, 0.329], [0.278, 0.176, 0.482], [0.231, 0.322, 0.545], [0.173, 0.447, 0.557], [0.129, 0.565, 0.549],
    [0.153, 0.678, 0.506], [0.365, 0.784, 0.388], [0.667, 0.863, 0.196], [0.992, 0.906, 0.145],
];
/// Jet, the classic rainbow map, 9 equally spaced anchors.
pub const JET: [[f32; 3]; 9] = [
    [0.0, 0.0, 0.5], [0.0, 0.0, 1.0], [0.0, 0.5, 1.0], [0.0, 1.0, 1.0], [0.5, 1.0, 0.5],
    [1.0, 1.0, 0.0], [1.0, 0.5, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0],
];
/// A diverging blue–white–red map.
pub const DIVERGING: [[f32; 3]; 3] = [[0.23, 0.30, 0.75], [1.0, 1.0, 1.0], [0.71, 0.02, 0.15]];

/// sRGB decoding (from the unit on RGB): stored value in [0, 1] → linear light.
pub fn srgb_decode(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB encoding: linear light in [0, 1] → stored value.
pub fn srgb_encode(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.0031308 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// C1a. The position t in [0, 1] of the data value `v` in the display range `lo..hi`:
/// 0 at `lo`, 1 at `hi`, clipped outside.
pub fn position(v: f32, lo: f32, hi: f32) -> f32 {
    todo!("C1a: position")
}

/// C1b. A colour map: every pixel becomes the colour at its position t, interpolated linearly
/// between the two nearest of the equally spaced `anchors` (anchor 0 at t = 0, the last at t = 1).
///
/// Hint: with n anchors, `s = t · (n − 1)`, `i = floor(s)` (at most n − 2), `f = s − i`;
/// the colour is `(1 − f) · anchors[i] + f · anchors[i + 1]`.
pub fn colour_map(img: &Image<f32>, lo: f32, hi: f32, anchors: &[[f32; 3]]) -> Rgb {
    todo!("C1b: colour_map")
}

/// C2a. The lightness L* (0 to 100) of an encoded sRGB colour: decode, the luminance
/// `Y = 0.2126 R + 0.7152 G + 0.0722 B`, then `L* = 116 f(Y) − 16` (the white has Y = 1).
///
/// Hint: f(t) = t^(1/3) above (6/29)³, else t / (3 (6/29)²) + 4/29, as in the unit on CIELAB.
pub fn lightness(c: [f32; 3]) -> f32 {
    todo!("C2a: lightness")
}

/// C2b. Is the sequence strictly increasing or strictly decreasing (a monotone lightness
/// profile)?
pub fn is_monotone(values: &[f32]) -> bool {
    todo!("C2b: is_monotone")
}

/// C3. Merge grey channels into one colour image: channel k (values in [0, 1]) times its
/// colour `colours[k]`, added up, clipped to [0, 1].
pub fn merge(channels: &[Image<f32>], colours: &[[f32; 3]]) -> Rgb {
    todo!("C3: merge")
}

/// C4. The deuteranope's matrix in LMS (slide "The math: a dichromat is a projection"):
/// rows (1, 0, 0), (a, 0, b), (0, 0, 1), with a and b chosen so that the two anchor colours
/// `white` and `blue` (both in LMS) keep their M: `a·L + b·S = M` for both.
///
/// Hint: two equations, two unknowns: solve with the determinant.
pub fn deutan_lms(white: [f32; 3], blue: [f32; 3]) -> [[f32; 3]; 3] {
    todo!("C4: deutan_lms")
}

/// C5a. The factors of a decorrelation stretch (slide "The math: decorrelation stretch"):
/// `(σ̄ / sqrt(λ_k))^amount` with `σ̄² = (λ1 + λ2 + λ3) / 3`, for the three eigenvalues of the
/// colour covariance; `amount` 0 = no stretch, 1 = full stretch.
pub fn stretch_factors(lambda: [f32; 3], amount: f32) -> [f32; 3] {
    todo!("C5a: stretch_factors")
}

/// C5b. Decorrelation stretch of a colour image: the mean colour μ and the 3 × 3 covariance
/// S of all pixels (divide by n), its eigenvectors U and eigenvalues λ, then every pixel
/// `x' = μ + U diag(factors) Uᵀ (x − μ)`. Values are **not** clipped.
///
/// Hint: `linalg::sym_eigen(&s)` returns `(values, vectors)` with `vectors[k]` the k-th
/// eigenvector (in f64; the matrix is `Vec<Vec<f64>>`).
pub fn decorrelation_stretch(img: &Rgb, amount: f32) -> Rgb {
    todo!("C5b: decorrelation_stretch")
}
