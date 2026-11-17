//! Helpers shared by the units of day 27 (used by the unit programs, not part of the exercises).

use imgcourse::Image;

/// The true labels of the test scene, `size × size`: a disk of radius 30 and an arm 3 pixels
/// wide running up to the right.
pub fn disk_with_arm(size: usize) -> Image<u32> {
    Image::from_fn(size, size, |x, y| {
        let (dx, dy) = (x as f32 - 54.0, y as f32 - 70.0);
        let disk = dx * dx + dy * dy <= 30.0 * 30.0;
        // distance to the line through the centre with direction (1, −1)/√2, beyond the disk
        let along = (dx - dy) / 2f32.sqrt();
        let across = (dx + dy) / 2f32.sqrt();
        let arm = along > 0.0 && along < 85.0 && across.abs() <= 1.5;
        if disk || arm { 1 } else { 0 }
    })
}
