// Model solution for "Regularisation and data augmentation".

use imgcourse::filters::{bilinear, gaussian_blur};
use imgcourse::{Border, Image, Rng};

/// C1. The mirror image, left to right: the output pixel (x, y) is the input pixel (w - 1 - x, y).
pub fn flip_horizontal(img: &Image<f32>) -> Image<f32> {
    let w = img.width();
    Image::from_fn(w, img.height(), |x, y| img[(w - 1 - x, y)])
}

/// C2. Rotation by `degrees` about the image centre ((w - 1) / 2, (h - 1) / 2), clockwise on
/// screen for positive angles, with bilinear interpolation and mirrored borders. For every
/// output pixel, turn its offset from the centre back by the angle and read the input there.
pub fn rotate(img: &Image<f32>, degrees: f32) -> Image<f32> {
    let (s, c) = degrees.to_radians().sin_cos();
    let (cx, cy) = ((img.width() as f32 - 1.0) / 2.0, (img.height() as f32 - 1.0) / 2.0);
    Image::from_fn(img.width(), img.height(), |x, y| {
        let (dx, dy) = (x as f32 - cx, y as f32 - cy);
        // the inverse rotation (anticlockwise by the angle) of the offset
        let (sx, sy) = (c * dx + s * dy, -s * dx + c * dy);
        bilinear(img, cx + sx, cy + sy, Border::Mirror)
    })
}

/// C3. Elastic deformation: two random displacement fields u and v (one normal number per
/// pixel, blurred with a Gaussian of `sigma` pixels, then scaled so that their root mean square
/// is `alpha` pixels). The output pixel (x, y) reads the input at (x + u, y + v), bilinear,
/// mirrored borders. The same `seed` gives the same deformation.
pub fn elastic(img: &Image<f32>, alpha: f32, sigma: f32, seed: u64) -> Image<f32> {
    let mut rng = Rng::new(seed);
    let (w, h) = (img.width(), img.height());
    let mut field = || {
        let noise = Image::from_fn(w, h, |_, _| rng.normal());
        let smooth = gaussian_blur(&noise, sigma);
        let rms = (smooth.pixels().iter().map(|v| v * v).sum::<f32>() / (w * h) as f32).sqrt();
        smooth.map(|v| alpha * v / rms)
    };
    let u = field();
    let v = field();
    Image::from_fn(w, h, |x, y| bilinear(img, x as f32 + u[(x, y)], y as f32 + v[(x, y)], Border::Mirror))
}

/// C4. A change of brightness and contrast: `gain * v^gamma + offset` for every grey value v
/// (negative values are first set to 0, so that the power is defined).
pub fn intensity(img: &Image<f32>, gain: f32, offset: f32, gamma: f32) -> Image<f32> {
    img.map(|v| gain * v.max(0.0).powf(gamma) + offset)
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
pub fn augment(img: &Image<f32>, a: &Augment, rng: &mut Rng) -> Image<f32> {
    let mut out = img.clone();
    if a.flips {
        if rng.uniform() < 0.5 {
            out = flip_horizontal(&out);
        }
        if rng.uniform() < 0.5 {
            out = flip_horizontal(&rotate(&out, 180.0));
        }
    }
    let angle = a.max_angle * (2.0 * rng.uniform() - 1.0);
    if a.max_angle > 0.0 {
        out = rotate(&out, angle);
    }
    let gain = 1.0 + a.gain_range * (2.0 * rng.uniform() - 1.0);
    out = intensity(&out, gain, 0.0, 1.0);
    if a.noise > 0.0 {
        out = out.map(|v| v + a.noise * rng.normal());
    }
    out
}
