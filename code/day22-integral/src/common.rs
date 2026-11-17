//! Helpers shared by the units of day 22: the integral image of the first unit and rectangle
//! sums from it (given, not part of the exercises).

use imgcourse::Image;

/// The integral image of the first unit (given): `table[(x, y)]` is the sum of `f(v)` over the
/// pixels `v` in columns `0..x` and rows `0..y`, so row 0 and column 0 are zero.
/// `integral(&img, |v| v)` sums the pixels, `integral(&img, |v| v * v)` their squares.
pub fn integral(img: &Image<f32>, f: impl Fn(f64) -> f64) -> Image<f64> {
    let mut s = Image::new(img.width() + 1, img.height() + 1, 0.0);
    for (x, y) in img.coords() {
        s[(x + 1, y + 1)] = f(img[(x, y)] as f64) + s[(x, y + 1)] + s[(x + 1, y)] - s[(x, y)];
    }
    s
}

/// The sum over the `w × h` rectangle whose top-left pixel is `(x, y)` (given): four lookups.
pub fn rect(s: &Image<f64>, x: usize, y: usize, w: usize, h: usize) -> f64 {
    s[(x + w, y + h)] - s[(x, y + h)] - s[(x + w, y)] + s[(x, y)]
}
