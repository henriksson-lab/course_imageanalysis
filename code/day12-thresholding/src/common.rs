//! Helpers shared by the units of day 12 (not exercises): the histogram bin of a grey value,
//! and showing a mask as a picture in the programs (`run.rs`).

use imgcourse::Image;

/// The bin of a grey value `v` in a histogram with `bins` equal bins over [0, 1]: bin `i`
/// holds the values from `i / bins` up to (not including) `(i + 1) / bins`. The value 1 goes
/// into the last bin, values outside [0, 1] into the first or last bin.
/// With 256 bins, an 8-bit grey level `k` (stored as `k / 255`) lands in bin `k`.
pub fn bin_of(v: f32, bins: usize) -> usize {
    let b = (v * bins as f32).floor();
    if b < 0.0 {
        0
    } else {
        (b as usize).min(bins - 1)
    }
}

/// A mask as a grey picture: white where true, black where false.
pub fn to_gray(mask: &Image<bool>) -> Image<f32> {
    mask.map(|b| if b { 1.0 } else { 0.0 })
}
