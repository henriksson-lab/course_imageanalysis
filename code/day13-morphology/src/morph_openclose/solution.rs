// Model solution for the coding exercises of "Opening, closing and the distance transform".
//
// A binary image is an `Image<bool>`: true = object, false = background. A structuring
// element (SE) is a list of offsets (dx, dy) from its origin. The border convention is
// "ignored" (slide "The math: why the opening is idempotent"): erosion treats the outside
// as object, dilation treats it as background.

use imgcourse::Image;

/// The 3 × 3 square (given, in `src/common.rs`).
pub use crate::common::SQUARE;

/// A digital disc of radius `r`: all offsets with dx² + dy² ≤ r² (given, in `src/common.rs`).
pub use crate::common::disc;

/// The diamond of radius `r`: all offsets with |dx| + |dy| ≤ r (the disc of the city-block distance).
pub fn diamond(r: usize) -> Vec<(isize, isize)> {
    let r = r as isize;
    let mut se = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            if dx.abs() + dy.abs() <= r {
                se.push((dx, dy));
            }
        }
    }
    se
}

/// A horizontal line of `length` pixels, starting at the origin.
pub fn line(length: usize) -> Vec<(isize, isize)> {
    (0..length as isize).map(|dx| (dx, 0)).collect()
}

/// Erosion (from the last unit): keep x where A(x + b) is object for every b; the outside
/// counts as object.
pub fn erode(a: &Image<bool>, se: &[(isize, isize)]) -> Image<bool> {
    a.map_xy(|x, y, _| {
        se.iter().all(|&(dx, dy)| {
            let (u, v) = (x as isize + dx, y as isize + dy);
            !a.contains(u, v) || a[(u as usize, v as usize)]
        })
    })
}

/// Dilation (from the last unit): x is object if A(x − b) is object for some b; the outside
/// counts as background.
pub fn dilate(a: &Image<bool>, se: &[(isize, isize)]) -> Image<bool> {
    a.map_xy(|x, y, _| {
        se.iter().any(|&(dx, dy)| {
            let (u, v) = (x as isize - dx, y as isize - dy);
            a.contains(u, v) && a[(u as usize, v as usize)]
        })
    })
}

/// C1a. Opening: erode, then dilate.
pub fn open(a: &Image<bool>, se: &[(isize, isize)]) -> Image<bool> {
    dilate(&erode(a, se), se)
}

/// C1b. Closing: dilate, then erode.
pub fn close(a: &Image<bool>, se: &[(isize, isize)]) -> Image<bool> {
    erode(&dilate(a, se), se)
}

/// C2. Granulometry: open `a` with each SE of the list (growing sizes) and return the area
/// left `area[k]` = the number of pixels of the opening by `ses[k]`, and the pattern
/// spectrum `spectrum[k] = area[k] − area[k + 1]` (one value fewer). It is signed: digital
/// discs are not exactly unions of smaller digital discs, so a step can be slightly negative.
pub fn granulometry(a: &Image<bool>, ses: &[Vec<(isize, isize)>]) -> (Vec<usize>, Vec<isize>) {
    let area: Vec<usize> = ses.iter().map(|se| open(a, se).pixels().iter().filter(|&&p| p).count()).collect();
    let spectrum = (0..area.len() - 1).map(|k| area[k] as isize - area[k + 1] as isize).collect();
    (area, spectrum)
}

/// "Infinitely far": larger than any distance in an image, small enough to add 2 to.
pub const FAR: u32 = u32::MAX / 2;

/// C3. The distance transform in two raster passes (slide "The math: two passes are
/// enough"): 0 on the background, the number of steps to the nearest background pixel on
/// the object. Steps go to the 4 neighbours (city-block) or, if `chessboard`, to all 8.
/// Only background inside the image counts; an object without any background stays at `FAR`.
pub fn distance_transform(a: &Image<bool>, chessboard: bool) -> Image<u32> {
    let mut d = a.map(|p| if p { FAR } else { 0 });
    // the neighbours that the forward pass has already visited: left and up (chessboard:
    // also up-left and up-right); the backward pass uses the mirrored ones
    let forward: Vec<(isize, isize)> = if chessboard { vec![(-1, 0), (-1, -1), (0, -1), (1, -1)] } else { vec![(-1, 0), (0, -1)] };
    let backward: Vec<(isize, isize)> = forward.iter().map(|&(dx, dy)| (-dx, -dy)).collect();
    for y in 0..a.height() {
        for x in 0..a.width() {
            relax(&mut d, x, y, &forward);
        }
    }
    for y in (0..a.height()).rev() {
        for x in (0..a.width()).rev() {
            relax(&mut d, x, y, &backward);
        }
    }
    d
}

/// d(x, y) = min(d(x, y), d(neighbour) + 1) over the given neighbours inside the image.
fn relax(d: &mut Image<u32>, x: usize, y: usize, steps: &[(isize, isize)]) {
    for &(dx, dy) in steps {
        let (u, v) = (x as isize + dx, y as isize + dy);
        if d.contains(u, v) {
            let through = d[(u as usize, v as usize)] + 1;
            if through < d[(x, y)] {
                d[(x, y)] = through;
            }
        }
    }
}

/// C4. Erosion by the diamond (city-block) or the square (chessboard) of radius `r` as a
/// threshold of the distance transform: object where DT > r.
pub fn erode_by_distance(a: &Image<bool>, r: u32, chessboard: bool) -> Image<bool> {
    distance_transform(a, chessboard).map(|d| d > r)
}
