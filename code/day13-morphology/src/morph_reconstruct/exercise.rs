// Coding exercises for "Reconstruction and h-extrema". Replace each `todo!()` with your
// code, then check it with `cargo test -p day13 morph_reconstruct::` and look at
// `cargo run -p day13 -- morph-reconstruct`.
#![allow(unused_variables, unused_imports)]
//
// Grey images are `Image<f32>`; a binary mask is an `Image<bool>` and is turned into
// 0.0 / 1.0 to use the same reconstruction. One step looks at the 3 × 3 square (8
// neighbours, `eight = true`) or at the cross (4 neighbours); in 1D (images of height 1)
// both are the window of 3. Pixels outside the image are not used.

use imgcourse::Image;

/// The offsets of one step: the 3 × 3 square, or the 5-pixel cross.
pub fn neighbourhood(eight: bool) -> Vec<(isize, isize)> {
    if eight {
        (-1..=1).flat_map(|dy| (-1..=1).map(move |dx| (dx, dy))).collect()
    } else {
        vec![(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)]
    }
}

/// A mask as grey values: 1.0 for true, 0.0 for false (given, in `src/common.rs`).
pub use crate::common::to_f32;

/// K1. Reconstruction by dilation of `marker` under `mask` (marker ≤ mask): repeat the
/// geodesic dilation M ← min(max of M over the neighbourhood, mask) until nothing changes.
/// Returns the result and the number of geodesic dilations that changed something.
///
/// Hint: in a `loop`, make the next marker as a **new** image with `m.map_xy(...)` (every pixel
/// must read the old marker), using the offsets of `neighbourhood(eight)` and skipping those
/// outside the image (`m.contains`). Stop when `next == m`.
pub fn reconstruct(marker: &Image<f32>, mask: &Image<f32>, eight: bool) -> (Image<f32>, usize) {
    todo!("K1: reconstruct")
}

/// K2. Reconstruction by erosion of `marker` above `mask` (marker ≥ mask): repeat
/// M ← max(min of M over the neighbourhood, mask) until nothing changes.
///
/// Hint: the same with min and max swapped; or use `reconstruct` on −marker and −mask, and negate the result.
pub fn reconstruct_by_erosion(marker: &Image<f32>, mask: &Image<f32>, eight: bool) -> (Image<f32>, usize) {
    todo!("K2: reconstruct_by_erosion")
}

/// K3a. The h-maxima transform: reconstruct I − h under I (8 neighbours).
pub fn h_maxima(img: &Image<f32>, h: f32) -> Image<f32> {
    todo!("K3a: h_maxima")
}

/// K3b. The h-minima transform: reconstruct I + h above I by erosion (8 neighbours).
pub fn h_minima(img: &Image<f32>, h: f32) -> Image<f32> {
    todo!("K3b: h_minima")
}

/// K3c. The regional maxima (plateaus whose neighbours are all lower), for images whose
/// grey levels differ by at least `step` where they differ: the pixels where
/// I − h_maxima(I, step) > 0. (The h-maxima with a tiny h lowers only the tops.)
///
/// Hint: compare with `step / 2.0` rather than with 0, to be safe from rounding.
pub fn regional_maxima(img: &Image<f32>, step: f32) -> Image<bool> {
    todo!("K3c: regional_maxima")
}

/// The regional minima: the regional maxima of −I.
pub fn regional_minima(img: &Image<f32>, step: f32) -> Image<bool> {
    regional_maxima(&img.map(|v| -v), step)
}

/// Is (x, y) on the frame, the outermost rows and columns of the image?
pub fn on_frame(a: &Image<bool>, x: usize, y: usize) -> bool {
    x == 0 || y == 0 || x == a.width() - 1 || y == a.height() - 1
}

/// K4a. Fill holes: flood the background from the background pixels on the frame
/// (4 neighbours); every background pixel that the flood does not reach is a hole and
/// becomes object.
///
/// Hint: seeds = the background (as 1.0) on the frame, 0.0 elsewhere; mask = the background as 1.0.
/// After `reconstruct(.., false)` a hole is a background pixel that is still 0.0.
pub fn fill_holes(a: &Image<bool>) -> Image<bool> {
    todo!("K4a: fill_holes")
}

/// K4b. Clear border: remove every object (8 neighbours) that touches the frame.
pub fn clear_border(a: &Image<bool>) -> Image<bool> {
    todo!("K4b: clear_border")
}

/// K5. Minima imposition (slide "Minima imposition"): M = 0 on the markers and max I + 1
/// elsewhere, G = min(I + 1, M); reconstruct M above G by erosion (8 neighbours). The result
/// has its regional minima exactly at the markers.
///
/// Hint: `img.max()` is the largest grey value.
pub fn impose_minima(img: &Image<f32>, markers: &Image<bool>) -> Image<f32> {
    todo!("K5: impose_minima")
}
