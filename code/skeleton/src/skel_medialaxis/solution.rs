// Model solution for the coding exercises of "Medial axis, distance transform and reconstruction".

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
pub fn distance_transform(mask: &Image<bool>, metric: Metric) -> Image<u32> {
    let (w, h) = (mask.width(), mask.height());
    let big = (w + h) as u32;
    let mut dt = mask.map(|v| if v { big } else { 0 });
    // forwards: the neighbours above and to the left; backwards: below and to the right
    let forward: &[(isize, isize)] = match metric {
        Metric::CityBlock => &[(-1, 0), (0, -1)],
        Metric::Chessboard => &[(-1, 0), (-1, -1), (0, -1), (1, -1)],
    };
    for backwards in [false, true] {
        for i in 0..w * h {
            let i = if backwards { w * h - 1 - i } else { i };
            let (x, y) = (i % w, i / w);
            if dt[(x, y)] == 0 {
                continue;
            }
            for &(dx, dy) in forward {
                let (dx, dy) = if backwards { (-dx, -dy) } else { (dx, dy) };
                let n = dt.get_border(x as isize + dx, y as isize + dy, Border::Zero); // 0 outside
                dt[(x, y)] = dt[(x, y)].min(n + 1);
            }
        }
    }
    dt
}

/// C2. The medial axis: the object pixels `p` whose disc is maximal, i.e. no neighbour `q`
/// (one step away in the metric) has `DT(q) ≥ DT(p) + 1`.
pub fn medial_axis(dt: &Image<u32>, metric: Metric) -> Image<bool> {
    dt.map_xy(|x, y, d| {
        d > 0 && metric.steps().iter().all(|&(dx, dy)| dt.get_border(x as isize + dx, y as isize + dy, Border::Zero) < d + 1)
    })
}

/// C3. Reconstruction: the union of the discs of the axis pixels. A pixel `q` is covered by
/// the disc of axis pixel `p` if `dist(p, q) ≤ DT(p) − 1`.
pub fn reconstruct(axis: &Image<bool>, dt: &Image<u32>, metric: Metric) -> Image<bool> {
    let mut out = Image::new(axis.width(), axis.height(), false);
    for (x, y) in axis.coords() {
        if !axis[(x, y)] {
            continue;
        }
        let r = dt[(x, y)] as isize - 1;
        for dy in -r..=r {
            for dx in -r..=r {
                let (qx, qy) = (x as isize + dx, y as isize + dy);
                if metric.dist(dx, dy) as isize <= r && out.contains(qx, qy) {
                    out[(qx as usize, qy as usize)] = true;
                }
            }
        }
    }
    out
}

/// C4. Lantuéjoul's skeleton with the 3 × 3 cross (`cross` true) or square as `B`: for
/// r = 0, 1, 2, ...: `E` = the mask eroded r times, `S_r` = `E` minus its opening by `B`
/// (erode, then dilate). Stop when `E` is empty. Returns an image with `r + 1` on the
/// pixels of `S_r` and 0 elsewhere.
pub fn lantuejoul(mask: &Image<bool>, cross: bool) -> Image<u32> {
    let mut out = Image::new(mask.width(), mask.height(), 0);
    let mut eroded = mask.clone();
    let mut r = 0;
    while eroded.pixels().iter().any(|&v| v) {
        let opened = dilate(&erode(&eroded, cross), cross);
        for (x, y) in eroded.coords() {
            if eroded[(x, y)] && !opened[(x, y)] {
                out[(x, y)] = r + 1;
            }
        }
        eroded = erode(&eroded, cross);
        r += 1;
    }
    out
}

/// C5. Rebuild the shape from Lantuéjoul's skeleton: the union over r of `S_r` dilated r
/// times by `B` (`skeleton` holds `r + 1` on `S_r`, as returned by `lantuejoul`). Dilating
/// each `S_r` separately costs 0 + 1 + 2 + ... dilations; instead go from the largest r
/// down: start with the largest `S_r`, then repeat "dilate once, add the next smaller
/// `S_r`". Each `S_r` is then dilated exactly r times, and there are only as many
/// dilations as levels.
pub fn reconstruct_lantuejoul(skeleton: &Image<u32>, cross: bool) -> Image<bool> {
    let max = skeleton.pixels().iter().copied().max().unwrap_or(0);
    let mut out = skeleton.map(|v| v == max && max > 0);
    for k in (1..max).rev() {
        // one more dilation for everything so far, then add S_(k−1) (stored as k)
        out = dilate(&out, cross).zip_map(skeleton, |a, v| a || v == k);
    }
    out
}
