//! Helpers shared by the programs (`run.rs`) of the units of day 36.

use imgcourse::{io, Image};

/// The colour of group or component `k` (0, 1, 2, …): the label colours of `imgcourse::io`.
pub fn colour(k: usize) -> [f32; 3] {
    io::label_colours(&Image::new(1, 1, k as u32 + 1))[(0, 0)]
}
