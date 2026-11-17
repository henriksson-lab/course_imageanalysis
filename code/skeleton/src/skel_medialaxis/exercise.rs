// Coding exercises for "Medial axis, distance transform and reconstruction". Replace each
// `todo!()` with your code, then check it with `cargo test -p skeleton skel_medialaxis::` and look at
// `cargo run -p skeleton -- skel-medialaxis`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};

/// How distances are measured on the grid: city-block steps (left, right, up, down; discs
/// are diamonds) or chessboard steps (diagonals too; discs are squares).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    CityBlock,
    Chessboard,
}

impl Metric {
    /// The distance between two pixels.
    pub fn dist(self, dx: isize, dy: isize) -> u32 {
        match self {
            Metric::CityBlock => (dx.abs() + dy.abs()) as u32,
            Metric::Chessboard => dx.abs().max(dy.abs()) as u32,
        }
    }

    /// The neighbours one step away, as offsets `(dx, dy)`.
    pub fn steps(self) -> &'static [(isize, isize)] {
        match self {
            Metric::CityBlock => &[(1, 0), (-1, 0), (0, 1), (0, -1)],
            Metric::Chessboard => &[(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)],
        }
    }
}

/// Erosion of a binary image by the 3 × 3 cross (`cross` true) or square: a pixel stays if the
/// whole element, centred there, lies in the object. Outside the image is background.
/// (The morphology day; not an exercise here.)
pub fn erode(img: &Image<bool>, cross: bool) -> Image<bool> {
    let m = if cross { Metric::CityBlock } else { Metric::Chessboard };
    img.map_xy(|x, y, v| v && m.steps().iter().all(|&(dx, dy)| img.get_border(x as isize + dx, y as isize + dy, Border::Zero)))
}

/// Dilation by the 3 × 3 cross or square: a pixel is set if the element, centred there, hits
/// the object.
pub fn dilate(img: &Image<bool>, cross: bool) -> Image<bool> {
    let m = if cross { Metric::CityBlock } else { Metric::Chessboard };
    img.map_xy(|x, y, v| v || m.steps().iter().any(|&(dx, dy)| img.get_border(x as isize + dx, y as isize + dy, Border::Zero)))
}

/// C1. The distance transform: for every object pixel, the number of steps to the nearest
/// background pixel (1 on the boundary); 0 on the background. Outside the image is
/// background. Two passes: forwards (row by row from the top left) looking at the
/// neighbours already visited, then backwards looking at the others.
///
/// Hint: start with 0 on the background and a large number (e.g. width + height) on the
/// object. Forwards, the visited neighbours are left and above (chessboard: also up-left and
/// up-right); set `dt(p) = min(dt(p), dt(q) + 1)`. Backwards, the mirrored neighbours.
/// `dt.get_border(x, y, Border::Zero)` reads 0 outside the image.
pub fn distance_transform(mask: &Image<bool>, metric: Metric) -> Image<u32> {
    todo!("C1: distance_transform")
}

/// C2. The medial axis: the object pixels `p` whose disc is maximal, i.e. no neighbour `q`
/// (one step away in the metric) has `DT(q) ≥ DT(p) + 1`.
///
/// Hint: `metric.steps()` lists the neighbours one step away.
pub fn medial_axis(dt: &Image<u32>, metric: Metric) -> Image<bool> {
    todo!("C2: medial_axis")
}

/// C3. Reconstruction: the union of the discs of the axis pixels. A pixel `q` is covered by
/// the disc of axis pixel `p` if `dist(p, q) ≤ DT(p) − 1`.
///
/// Hint: for every axis pixel, loop over the square of offsets `-r..=r` with `r = DT − 1`
/// and keep those with `metric.dist(dx, dy) ≤ r` that are inside the image.
pub fn reconstruct(axis: &Image<bool>, dt: &Image<u32>, metric: Metric) -> Image<bool> {
    todo!("C3: reconstruct")
}

/// C4. Lantuéjoul's skeleton with the 3 × 3 cross (`cross` true) or square as `B`: for
/// r = 0, 1, 2, ...: `E` = the mask eroded r times, `S_r` = `E` minus its opening by `B`
/// (erode, then dilate). Stop when `E` is empty. Returns an image with `r + 1` on the
/// pixels of `S_r` and 0 elsewhere.
///
/// Hint: use the helpers `erode` and `dilate` above.
pub fn lantuejoul(mask: &Image<bool>, cross: bool) -> Image<u32> {
    todo!("C4: lantuejoul")
}

/// C5. Rebuild the shape from Lantuéjoul's skeleton: the union over r of `S_r` dilated r
/// times by `B` (`skeleton` holds `r + 1` on `S_r`, as returned by `lantuejoul`). Dilating
/// each `S_r` separately costs 0 + 1 + 2 + ... dilations; instead go from the largest r
/// down: start with the largest `S_r`, then repeat "dilate once, add the next smaller
/// `S_r`". Each `S_r` is then dilated exactly r times, and there are only as many
/// dilations as levels.
pub fn reconstruct_lantuejoul(skeleton: &Image<u32>, cross: bool) -> Image<bool> {
    todo!("C5: reconstruct_lantuejoul")
}
