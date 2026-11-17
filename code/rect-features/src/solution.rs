// Model solution for the coding exercises of "Rectangle features".

use imgcourse::Image;

/// The integral image (given, from the first unit of the day): `table[(x, y)]` is the sum of
/// the pixels in columns `0..x` and rows `0..y`; row 0 and column 0 are zero.
pub fn integral(img: &Image<f32>) -> Image<f64> {
    let mut s = Image::new(img.width() + 1, img.height() + 1, 0.0);
    for (x, y) in img.coords() {
        s[(x + 1, y + 1)] = img[(x, y)] as f64 + s[(x, y + 1)] + s[(x + 1, y)] - s[(x, y)];
    }
    s
}

/// The sum over the `w × h` rectangle whose top-left pixel is `(x, y)` (given): four lookups.
pub fn rect(s: &Image<f64>, x: usize, y: usize, w: usize, h: usize) -> f64 {
    s[(x + w, y + h)] - s[(x, y + h)] - s[(x + w, y)] + s[(x, y)]
}

// In all features below, `(x, y)` is the top-left pixel of the whole template and `w × h` the
// size of one of its rectangles.

/// C1. Edge across: the sum of the left rectangle minus the sum of the right one, two `w × h`
/// rectangles side by side (exercise B1).
pub fn edge_across(s: &Image<f64>, x: usize, y: usize, w: usize, h: usize) -> f64 {
    rect(s, x, y, w, h) - rect(s, x + w, y, w, h)
}

/// C1. Edge down: the top rectangle minus the bottom one, one above the other.
pub fn edge_down(s: &Image<f64>, x: usize, y: usize, w: usize, h: usize) -> f64 {
    rect(s, x, y, w, h) - rect(s, x, y + h, w, h)
}

/// C2. Line across: three rectangles side by side, `2 · centre − left − right` (exercise B2).
pub fn line_across(s: &Image<f64>, x: usize, y: usize, w: usize, h: usize) -> f64 {
    2.0 * rect(s, x + w, y, w, h) - rect(s, x, y, w, h) - rect(s, x + 2 * w, y, w, h)
}

/// C2. Line down: three rectangles one above the other, `2 · centre − top − bottom`.
pub fn line_down(s: &Image<f64>, x: usize, y: usize, w: usize, h: usize) -> f64 {
    2.0 * rect(s, x, y + h, w, h) - rect(s, x, y, w, h) - rect(s, x, y + 2 * h, w, h)
}

/// C3. Diagonal: four rectangles in a 2 × 2 block, top-left + bottom-right − top-right −
/// bottom-left.
pub fn diagonal(s: &Image<f64>, x: usize, y: usize, w: usize, h: usize) -> f64 {
    rect(s, x, y, w, h) + rect(s, x + w, y + h, w, h) - rect(s, x + w, y, w, h) - rect(s, x, y + h, w, h)
}

/// C3. Centre − surround: the template is `3w × 3h`; the sum of its centre `w × h` rectangle
/// minus the sum of the whole template divided by 9 (so that a flat region gives 0).
pub fn centre_surround(s: &Image<f64>, x: usize, y: usize, w: usize, h: usize) -> f64 {
    rect(s, x + w, y + h, w, h) - rect(s, x, y, 3 * w, 3 * h) / 9.0
}

/// C4. The number of ways to place a template of `a × b` rectangles (`a` across, `b` down) in a
/// `width × height` window: every rectangle size `w × h` (from 1 pixel) and every position
/// where the whole template fits (slide "The pool is huge").
pub fn count_placements(a: usize, b: usize, width: usize, height: usize) -> usize {
    let mut count = 0;
    for w in 1..=width / a {
        for h in 1..=height / b {
            count += (width - a * w + 1) * (height - b * h + 1);
        }
    }
    count
}
