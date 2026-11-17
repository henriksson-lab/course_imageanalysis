//! Helpers shared by the units of day 17 (used by the unit programs, not part of the exercises).

use imgcourse::Image;

/// The number of `true` pixels.
pub fn count(img: &Image<bool>) -> usize {
    img.pixels().iter().filter(|&&v| v).count()
}
