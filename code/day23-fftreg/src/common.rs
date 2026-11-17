//! Helpers shared by the units of day 23 (used by the unit programs, not part of the exercises).

use imgcourse::Image;

/// Moves shift 0 to the middle of the picture.
pub fn centred(c: &Image<f32>) -> Image<f32> {
    let (w, h) = (c.width(), c.height());
    Image::from_fn(w, h, |x, y| c[((x + w / 2) % w, (y + h / 2) % h)])
}
