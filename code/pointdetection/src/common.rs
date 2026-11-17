//! Helpers shared by the units of day 20: the synthetic spot images (not part of the exercises).

use imgcourse::{Image, Rng};

/// A synthetic spot image: `n` Gaussian spots of width `sigma` (pixels) and peak amplitudes
/// between `amp.0` and `amp.1`, on the background `background(x, y)`, with Gaussian noise of
/// standard deviation `noise`. Returns the image and the true spot centres `[x, y]`.
pub fn spot_scene(
    size: usize,
    n: usize,
    sigma: f32,
    amp: (f32, f32),
    noise: f32,
    background: impl Fn(f32, f32) -> f32,
    seed: u64,
) -> (Image<f32>, Vec<[f32; 2]>) {
    let mut rng = Rng::new(seed);
    let margin = 3.0 * sigma;
    let spots: Vec<([f32; 2], f32)> = (0..n)
        .map(|_| {
            let x = margin + (size as f32 - 2.0 * margin) * rng.uniform();
            let y = margin + (size as f32 - 2.0 * margin) * rng.uniform();
            ([x, y], amp.0 + (amp.1 - amp.0) * rng.uniform())
        })
        .collect();
    let mut img = Image::from_fn(size, size, |x, y| background(x as f32, y as f32));
    let reach = (4.0 * sigma).ceil() as isize;
    for &([sx, sy], a) in &spots {
        let (cx, cy) = (sx.round() as isize, sy.round() as isize);
        for y in cy - reach..=cy + reach {
            for x in cx - reach..=cx + reach {
                if img.contains(x, y) {
                    let d2 = (x as f32 - sx).powi(2) + (y as f32 - sy).powi(2);
                    img[(x as usize, y as usize)] += a * (-d2 / (2.0 * sigma * sigma)).exp();
                }
            }
        }
    }
    for v in img.pixels_mut() {
        *v += noise * rng.normal();
    }
    (img, spots.into_iter().map(|(p, _)| p).collect())
}
