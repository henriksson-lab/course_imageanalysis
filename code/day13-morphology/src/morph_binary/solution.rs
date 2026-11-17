// Model solution for the coding exercises of "Erosion and dilation".
//
// A binary image is an `Image<bool>`: true = object (1), false = background (0).
// A structuring element (SE) is a list of offsets (dx, dy) from its origin; x to the right,
// y down.

use imgcourse::Image;

/// The 5-pixel cross.
pub const CROSS: &[(isize, isize)] = &[(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)];
/// The 3 × 3 square (given, in `src/common.rs`).
pub use crate::common::SQUARE;
/// The L of the exercises: the origin, the pixel to its right and the pixel below it.
pub const L: &[(isize, isize)] = &[(0, 0), (1, 0), (0, 1)];

/// A digital disc of radius `r`: all offsets with dx² + dy² ≤ r² (given, in `src/common.rs`).
pub use crate::common::disc;

/// The reflected SE: every offset b replaced by −b.
pub fn reflect(se: &[(isize, isize)]) -> Vec<(isize, isize)> {
    se.iter().map(|&(dx, dy)| (-dx, -dy)).collect()
}

/// The complement: object and background swapped.
pub fn complement(a: &Image<bool>) -> Image<bool> {
    a.map(|v| !v)
}

/// The value of `a` at (x, y), or `outside` if (x, y) lies outside the image.
fn value(a: &Image<bool>, x: isize, y: isize, outside: bool) -> bool {
    if a.contains(x, y) {
        a[(x as usize, y as usize)]
    } else {
        outside
    }
}

/// C1. Erosion: keep the pixel x where the SE, placed with its origin on x, lies entirely
/// on the object: A(x + b) is true for every offset b. Outside the image the value
/// `outside` is assumed (slide "At the image border").
pub fn erode(a: &Image<bool>, se: &[(isize, isize)], outside: bool) -> Image<bool> {
    a.map_xy(|x, y, _| se.iter().all(|&(dx, dy)| value(a, x as isize + dx, y as isize + dy, outside)))
}

/// C2. Dilation: the pixel x is object if the reflected SE placed at x hits the object:
/// A(x − b) is true for some offset b. Outside the image the value `outside` is assumed.
pub fn dilate(a: &Image<bool>, se: &[(isize, isize)], outside: bool) -> Image<bool> {
    a.map_xy(|x, y, _| se.iter().any(|&(dx, dy)| value(a, x as isize - dx, y as isize - dy, outside)))
}

/// C3. The inner boundary: the object pixels that the erosion by `se` removes, A \ (A ⊖ B).
/// Outside the image counts as background.
pub fn inner_boundary(a: &Image<bool>, se: &[(isize, isize)]) -> Image<bool> {
    let eroded = erode(a, se, false);
    a.zip_map(&eroded, |p, e| p && !e)
}
