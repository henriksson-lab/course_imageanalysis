//! Helpers shared by the units of day 5 (not exercises).

use imgcourse::Image;

/// Root-mean-square difference of two images of the same size (summed in f32).
pub fn rmse(a: &Image<f32>, b: &Image<f32>) -> f32 {
    let s: f32 = a.pixels().iter().zip(b.pixels()).map(|(p, q)| (p - q).powi(2)).sum();
    (s / a.pixels().len() as f32).sqrt()
}
