// Coding exercises for "Generating and exploring images with a VAE". Replace each `todo!()` with your code,
// then check it with `cargo test -p day38 vae_gen::` and look at `cargo run -p day38 -- vae-gen`.
#![allow(unused_variables, unused_imports)]

// Instead of a trained network, the decoder here is a fixed formula that draws a soft
// bright spot (a "cell") whose position is the latent code z: a curved decoder with a 2D
// latent space, small enough to understand completely.

use imgcourse::{Image, Rng};

/// A decoder: turns a latent code `z` into an image.
pub type Decoder = dyn Fn([f32; 2]) -> Image<f32>;

/// The spot decoder: a `size × size` image with a soft spot (a Gaussian of sd 2 pixels)
/// centred at `(size/2 + 4 z1, size/2 + 4 z2)`. The prior N(0, I) puts most spots near the
/// centre.
pub fn spot_decoder(size: usize) -> impl Fn([f32; 2]) -> Image<f32> {
    move |z| {
        let c = size as f32 / 2.0;
        let (cx, cy) = (c + 4.0 * z[0], c + 4.0 * z[1]);
        Image::from_fn(size, size, |x, y| {
            let d2 = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
            (-d2 / 8.0).exp()
        })
    }
}

/// The bar decoder of the lecture: a 3 × 1 image where pixel `i` gets `max(0, 1 − |z1 − i|)`
/// (`z2` is not used).
pub fn bar_decoder(z: [f32; 2]) -> Image<f32> {
    Image::from_fn(3, 1, |i, _| (1.0 - (z[0] - i as f32).abs()).max(0.0))
}

/// C1a. A walk in latent space: decode `n` equally spaced points
/// `z(t) = (1 − t) za + t zb`, `t = 0, 1/(n−1), ..., 1`.
///
/// Hint: for k in 0..n, t = k / (n − 1); mix the two codes coordinate by coordinate and call
/// `decode(z)`.
pub fn latent_walk(decode: &Decoder, za: [f32; 2], zb: [f32; 2], n: usize) -> Vec<Image<f32>> {
    todo!("C1a: latent_walk")
}

/// C1b. The pixel fade between two images: `(1 − t) a + t b` for `n` equally spaced `t`.
///
/// Hint: `a.zip_map(b, |va, vb| ...)` mixes two images pixel by pixel.
pub fn pixel_fade(a: &Image<f32>, b: &Image<f32>, n: usize) -> Vec<Image<f32>> {
    todo!("C1b: pixel_fade")
}

/// C2. Draw `n` codes from the 2D prior N(0, I) and return the fraction that lands within
/// `radius` of 0.
///
/// Hint: `rng.normal()` twice gives a draw of z; it is inside when z1² + z2² ≤ radius².
pub fn fraction_within(radius: f32, n: usize, seed: u64) -> f32 {
    todo!("C2: fraction_within")
}

/// The mean squared pixel error between two images of the same size.
pub fn mean_squared_error(a: &Image<f32>, b: &Image<f32>) -> f32 {
    a.zip_map(b, |u, v| (u - v).powi(2)).mean()
}

/// C3. Encode by search: try every code on an `n × n` grid from `lo` to `hi` in both
/// coordinates, decode it, and return the code whose image is closest to `img` (smallest
/// mean squared error), with that error. (A VAE's encoder network learns to give this
/// answer in one pass; here we search.)
///
/// Hint: two loops over the grid; keep the best code and error so far, starting with the error
/// `f32::INFINITY`; `mean_squared_error` compares two images.
pub fn encode_by_search(img: &Image<f32>, decode: &Decoder, lo: f32, hi: f32, n: usize) -> ([f32; 2], f32) {
    todo!("C3: encode_by_search")
}

/// C4. The anomaly threshold: the smallest error such that a fraction `q` of the `errors`
/// of normal images is at or below it (sort, then take entry `ceil(q n) − 1`).
///
/// Hint: copy and sort the errors (`sort_by(|a, b| a.total_cmp(b))`); `(q * n as f32).ceil() as
/// usize` counts how many must be at or below the threshold.
pub fn threshold(errors: &[f32], q: f32) -> f32 {
    todo!("C4: threshold")
}
