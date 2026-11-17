// Coding exercises for "Erosion and dilation". Replace each `todo!()` with your code, then
// check it with `cargo test -p morph-binary` and look at `cargo run -p morph-binary`.
#![allow(unused_variables, unused_imports)]
//
// A binary image is an `Image<bool>`: true = object (1), false = background (0).
// A structuring element (SE) is a list of offsets (dx, dy) from its origin; x to the right,
// y down.

use imgcourse::Image;

/// The 5-pixel cross.
pub const CROSS: &[(isize, isize)] = &[(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)];
/// The 3 × 3 square.
pub const SQUARE: &[(isize, isize)] = &[(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];
/// The L of the exercises: the origin, the pixel to its right and the pixel below it.
pub const L: &[(isize, isize)] = &[(0, 0), (1, 0), (0, 1)];

/// A digital disc of radius `r`: all offsets with dx² + dy² ≤ r².
pub fn disc(r: usize) -> Vec<(isize, isize)> {
    let r = r as isize;
    let mut se = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                se.push((dx, dy));
            }
        }
    }
    se
}

/// The reflected SE: every offset b replaced by −b.
pub fn reflect(se: &[(isize, isize)]) -> Vec<(isize, isize)> {
    se.iter().map(|&(dx, dy)| (-dx, -dy)).collect()
}

/// The complement: object and background swapped.
pub fn complement(a: &Image<bool>) -> Image<bool> {
    a.map(|v| !v)
}

/// C1. Erosion: keep the pixel x where the SE, placed with its origin on x, lies entirely
/// on the object: A(x + b) is true for every offset b. Outside the image the value
/// `outside` is assumed (slide "At the image border").
///
/// Hint: `a.contains(x, y)` takes signed coordinates and tells whether (x, y) is inside.
/// For every pixel, loop over the offsets `for &(dx, dy) in se`.
pub fn erode(a: &Image<bool>, se: &[(isize, isize)], outside: bool) -> Image<bool> {
    todo!("C1: erode")
}

/// C2. Dilation: the pixel x is object if the reflected SE placed at x hits the object:
/// A(x − b) is true for some offset b. Outside the image the value `outside` is assumed.
///
/// Hint: the same loops as `erode`, with "some" instead of "every", and x − b instead of x + b.
pub fn dilate(a: &Image<bool>, se: &[(isize, isize)], outside: bool) -> Image<bool> {
    todo!("C2: dilate")
}

/// C3. The inner boundary: the object pixels that the erosion by `se` removes, A \ (A ⊖ B).
/// Outside the image counts as background.
pub fn inner_boundary(a: &Image<bool>, se: &[(isize, isize)]) -> Image<bool> {
    todo!("C3: inner_boundary")
}
