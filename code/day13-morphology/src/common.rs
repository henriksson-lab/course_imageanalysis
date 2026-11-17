//! Helpers shared by the units of day 13 (not exercises): the 3 × 3 square and the digital
//! disc as structuring elements, and masks as pictures and pixel counts.

use imgcourse::Image;

/// The 3 × 3 square.
pub const SQUARE: &[(isize, isize)] = &[(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];

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

/// A mask as grey values: 1.0 for true, 0.0 for false.
pub fn to_f32(a: &Image<bool>) -> Image<f32> {
    a.map(|b| if b { 1.0 } else { 0.0 })
}

/// The number of pixels that are true.
pub fn count(a: &Image<bool>) -> usize {
    a.pixels().iter().filter(|&&b| b).count()
}
