//! Simulated spots with Poisson noise, and enlarging patches for viewing.
//! Not part of the exercise: read it if you like.

use imgcourse::{Image, Rng};

/// The expected photon counts of a square patch: `photons` photons from a Gaussian spot of width
/// `s` (pixels) centred at `(cx, cy)`, plus `background` photons per pixel. Each pixel gets the
/// spot's density at its centre (point sampling).
pub fn spot_mean(size: usize, spots: &[(f32, f32, f32)], s: f32, background: f32) -> Image<f32> {
    let norm = 1.0 / (2.0 * std::f32::consts::PI * s * s);
    Image::from_fn(size, size, |x, y| {
        let mut v = background;
        for &(cx, cy, photons) in spots {
            let d2 = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
            v += photons * norm * (-d2 / (2.0 * s * s)).exp();
        }
        v
    })
}

/// Photon counts: every pixel of `mean` replaced by a Poisson count with that mean.
pub fn with_poisson(mean: &Image<f32>, rng: &mut Rng) -> Image<f32> {
    Image::from_fn(mean.width(), mean.height(), |x, y| rng.poisson(mean[(x, y)]))
}

/// The brightest pixel of a patch.
pub fn brightest(patch: &Image<f32>) -> (usize, usize) {
    patch.coords().max_by(|&a, &b| patch[a].total_cmp(&patch[b])).unwrap()
}

/// Every pixel repeated `f × f` times, to look at small patches.
pub fn enlarge(img: &Image<f32>, f: usize) -> Image<f32> {
    Image::from_fn(img.width() * f, img.height() * f, |x, y| img[(x / f, y / f)])
}
