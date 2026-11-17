// Model solution for the coding exercises of "Reconstruction and h-extrema".
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
pub fn reconstruct(marker: &Image<f32>, mask: &Image<f32>, eight: bool) -> (Image<f32>, usize) {
    let nb = neighbourhood(eight);
    let mut m = marker.clone();
    let mut iterations = 0;
    loop {
        // one geodesic dilation, into a new image (every pixel reads the old marker)
        let next = m.map_xy(|x, y, _| {
            let mut largest = f32::NEG_INFINITY;
            for &(dx, dy) in &nb {
                let (u, v) = (x as isize + dx, y as isize + dy);
                if m.contains(u, v) {
                    largest = largest.max(m[(u as usize, v as usize)]);
                }
            }
            largest.min(mask[(x, y)])
        });
        if next == m {
            return (m, iterations);
        }
        m = next;
        iterations += 1;
    }
}

/// K2. Reconstruction by erosion of `marker` above `mask` (marker ≥ mask): repeat
/// M ← max(min of M over the neighbourhood, mask) until nothing changes.
pub fn reconstruct_by_erosion(marker: &Image<f32>, mask: &Image<f32>, eight: bool) -> (Image<f32>, usize) {
    // by duality: erosion above `mask` is dilation of −M under −mask
    let (r, iterations) = reconstruct(&marker.map(|v| -v), &mask.map(|v| -v), eight);
    (r.map(|v| -v), iterations)
}

/// K3a. The h-maxima transform: reconstruct I − h under I (8 neighbours).
pub fn h_maxima(img: &Image<f32>, h: f32) -> Image<f32> {
    reconstruct(&img.map(|v| v - h), img, true).0
}

/// K3b. The h-minima transform: reconstruct I + h above I by erosion (8 neighbours).
pub fn h_minima(img: &Image<f32>, h: f32) -> Image<f32> {
    reconstruct_by_erosion(&img.map(|v| v + h), img, true).0
}

/// K3c. The regional maxima (plateaus whose neighbours are all lower), for images whose
/// grey levels differ by at least `step` where they differ: the pixels where
/// I − h_maxima(I, step) > 0. (The h-maxima with a tiny h lowers only the tops.)
pub fn regional_maxima(img: &Image<f32>, step: f32) -> Image<bool> {
    img.zip_map(&h_maxima(img, step), |i, r| i - r > step / 2.0)
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
pub fn fill_holes(a: &Image<bool>) -> Image<bool> {
    let background = to_f32(&a.map(|p| !p));
    let seeds = background.map_xy(|x, y, v| if on_frame(a, x, y) { v } else { 0.0 });
    let (reached, _) = reconstruct(&seeds, &background, false);
    a.zip_map(&reached, |p, r| p || r == 0.0)
}

/// K4b. Clear border: remove every object (8 neighbours) that touches the frame.
pub fn clear_border(a: &Image<bool>) -> Image<bool> {
    let object = to_f32(a);
    let seeds = object.map_xy(|x, y, v| if on_frame(a, x, y) { v } else { 0.0 });
    let (touching, _) = reconstruct(&seeds, &object, true);
    a.zip_map(&touching, |p, t| p && t == 0.0)
}

/// K5. Minima imposition (slide "Minima imposition"): M = 0 on the markers and max I + 1
/// elsewhere, G = min(I + 1, M); reconstruct M above G by erosion (8 neighbours). The result
/// has its regional minima exactly at the markers.
pub fn impose_minima(img: &Image<f32>, markers: &Image<bool>) -> Image<f32> {
    let high = img.max() + 1.0;
    let m = markers.map(|is_marker| if is_marker { 0.0 } else { high });
    let g = img.zip_map(&m, |i, mv| (i + 1.0).min(mv));
    reconstruct_by_erosion(&m, &g, true).0
}
