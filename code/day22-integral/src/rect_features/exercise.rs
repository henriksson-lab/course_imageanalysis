// Coding exercises for "Rectangle features". Replace each `todo!()` with your code,
// then check it with `cargo test -p day22 rect_features::` and look at `cargo run -p day22 -- rect-features`.
#![allow(unused_variables, unused_imports)]

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
///
/// Hint: two calls of `rect`; the right rectangle starts at `x + w`. (With the table
/// entries written out, the two shared corners are read once: 6 lookups.)
pub fn edge_across(s: &Image<f64>, x: usize, y: usize, w: usize, h: usize) -> f64 {
    todo!("C1: edge_across")
}

/// C1. Edge down: the top rectangle minus the bottom one, one above the other.
pub fn edge_down(s: &Image<f64>, x: usize, y: usize, w: usize, h: usize) -> f64 {
    todo!("C1: edge_down")
}

/// C2. Line across: three rectangles side by side, `2 · centre − left − right` (exercise B2).
///
/// Hint: the centre rectangle starts at `x + w`, the right one at `x + 2 * w`.
pub fn line_across(s: &Image<f64>, x: usize, y: usize, w: usize, h: usize) -> f64 {
    todo!("C2: line_across")
}

/// C2. Line down: three rectangles one above the other, `2 · centre − top − bottom`.
pub fn line_down(s: &Image<f64>, x: usize, y: usize, w: usize, h: usize) -> f64 {
    todo!("C2: line_down")
}

/// C3. Diagonal: four rectangles in a 2 × 2 block, top-left + bottom-right − top-right −
/// bottom-left.
///
/// Hint: the four rectangles start at `(x, y)`, `(x + w, y)`, `(x, y + h)` and `(x + w, y + h)`.
pub fn diagonal(s: &Image<f64>, x: usize, y: usize, w: usize, h: usize) -> f64 {
    todo!("C3: diagonal")
}

/// C3. Centre − surround: the template is `3w × 3h`; the sum of its centre `w × h` rectangle
/// minus the sum of the whole template divided by 9 (so that a flat region gives 0).
///
/// Hint: the whole template is one rectangle of `3 * w × 3 * h` at `(x, y)`.
pub fn centre_surround(s: &Image<f64>, x: usize, y: usize, w: usize, h: usize) -> f64 {
    todo!("C3: centre_surround")
}

/// C4. The number of ways to place a template of `a × b` rectangles (`a` across, `b` down) in a
/// `width × height` window: every rectangle size `w × h` (from 1 pixel) and every position
/// where the whole template fits (slide "The pool is huge").
///
/// Hint: the template covers `a * w × b * h` pixels; across, it fits at `width - a * w + 1`
/// positions.
pub fn count_placements(a: usize, b: usize, width: usize, height: usize) -> usize {
    todo!("C4: count_placements")
}
