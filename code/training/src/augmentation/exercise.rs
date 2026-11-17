// Coding exercises for "Regularisation and data augmentation". Replace each `todo!()` with your
// code, then check it with `cargo test -p training augmentation::` and look at
// `cargo run -p training -- augmentation`.
#![allow(unused_variables, unused_imports)]

use imgcourse::filters::{bilinear, gaussian_blur};
use imgcourse::{Border, Image, Rng};

/// C1. The mirror image, left to right: the output pixel (x, y) is the input pixel (w - 1 - x, y).
///
/// Hint: `Image::from_fn(w, h, |x, y| ...)`.
pub fn flip_horizontal(img: &Image<f32>) -> Image<f32> {
    todo!("C1: flip_horizontal")
}

/// C2. Rotation by `degrees` about the image centre ((w - 1) / 2, (h - 1) / 2), clockwise on
/// screen for positive angles, with bilinear interpolation and mirrored borders. For every
/// output pixel, turn its offset from the centre back by the angle and read the input there.
///
/// Hint: with `(s, c) = degrees.to_radians().sin_cos()` and the offset (dx, dy) of the output
/// pixel, read the input at the centre plus `(c dx + s dy, -s dx + c dy)`, with
/// `bilinear(img, x, y, Border::Mirror)`.
pub fn rotate(img: &Image<f32>, degrees: f32) -> Image<f32> {
    todo!("C2: rotate")
}

/// C3. Elastic deformation: two random displacement fields u and v (one normal number per
/// pixel, blurred with a Gaussian of `sigma` pixels, then scaled so that their root mean square
/// is `alpha` pixels). The output pixel (x, y) reads the input at (x + u, y + v), bilinear,
/// mirrored borders. The same `seed` gives the same deformation.
///
/// Hint: one `Rng::new(seed)`; make u first, then v, each as `Image::from_fn(w, h, |_, _| rng.normal())`
/// passed through `gaussian_blur(&noise, sigma)`, then multiplied by `alpha / rms`.
pub fn elastic(img: &Image<f32>, alpha: f32, sigma: f32, seed: u64) -> Image<f32> {
    todo!("C3: elastic")
}

/// C4. A change of brightness and contrast: `gain * v^gamma + offset` for every grey value v
/// (negative values are first set to 0, so that the power is defined).
///
/// Hint: `img.map(|v| ...)`, `v.max(0.0).powf(gamma)`.
pub fn intensity(img: &Image<f32>, gain: f32, offset: f32, gamma: f32) -> Image<f32> {
    todo!("C4: intensity")
}

/// Which random changes `augment` makes.
#[derive(Clone, Copy, Debug)]
pub struct Augment {
    /// Flip left-right with probability 1/2, and up-down with probability 1/2.
    pub flips: bool,
    /// Rotate by a random angle between -max_angle and +max_angle degrees.
    pub max_angle: f32,
    /// Multiply the grey values by a random gain between 1 - gain_range and 1 + gain_range.
    pub gain_range: f32,
    /// Add Gaussian noise with this standard deviation.
    pub noise: f32,
}

/// C5. One random augmented copy: in this order, the flips (left-right: `flip_horizontal`;
/// up-down: rotate by 180 degrees and flip left-right), the rotation, the gain (`intensity`
/// with offset 0 and gamma 1), the noise. Draw the random numbers from `rng` in this order: one
/// uniform number per flip (flip if below 0.5), one for the angle, one for the gain, then one
/// normal number per pixel.
///
/// Hint: the angle is `max_angle * (2u - 1)` for a uniform u, the gain `1 + gain_range * (2u - 1)`.
/// Draw the angle's and the gain's numbers even when the range is 0 (so the order stays the
/// same), but skip the rotation when `max_angle` is 0 and the noise when `noise` is 0.
pub fn augment(img: &Image<f32>, a: &Augment, rng: &mut Rng) -> Image<f32> {
    todo!("C5: augment")
}
