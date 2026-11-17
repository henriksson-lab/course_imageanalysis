// Coding exercises for "Opening, closing and the distance transform". Replace each `todo!()`
// with your code, then check it with `cargo test -p day13 morph_openclose::` and look at
// `cargo run -p day13 -- morph-openclose`.
#![allow(unused_variables, unused_imports)]
//
// A binary image is an `Image<bool>`: true = object, false = background. A structuring
// element (SE) is a list of offsets (dx, dy) from its origin. The border convention is
// "ignored" (slide "The math: why the opening is idempotent"): erosion treats the outside
// as object, dilation treats it as background.

use imgcourse::Image;

/// The 3 × 3 square.
pub const SQUARE: &[(isize, isize)] = &[(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];

/// A digital disc of radius `r`: all offsets with dx² + dy² ≤ r².
pub fn disc(r: usize) -> Vec<(isize, isize)> {
    let r = r as isize;
    let mut se = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                se.push((dx, dy));
            }
        }
    }
    se
}

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
///
/// Hint: use `erode` and `dilate` above.
pub fn open(a: &Image<bool>, se: &[(isize, isize)]) -> Image<bool> {
    todo!("C1a: open")
}

/// C1b. Closing: dilate, then erode.
pub fn close(a: &Image<bool>, se: &[(isize, isize)]) -> Image<bool> {
    todo!("C1b: close")
}

/// C2. Granulometry: open `a` with each SE of the list (growing sizes) and return the area
/// left `area[k]` = the number of pixels of the opening by `ses[k]`, and the pattern
/// spectrum `spectrum[k] = area[k] − area[k + 1]` (one value fewer). It is signed: digital
/// discs are not exactly unions of smaller digital discs, so a step can be slightly negative.
///
/// Hint: count the object pixels with `.pixels().iter().filter(|&&p| p).count()`.
pub fn granulometry(a: &Image<bool>, ses: &[Vec<(isize, isize)>]) -> (Vec<usize>, Vec<isize>) {
    todo!("C2: granulometry")
}

/// "Infinitely far": larger than any distance in an image, small enough to add 2 to.
pub const FAR: u32 = u32::MAX / 2;

/// C3. The distance transform in two raster passes (slide "The math: two passes are
/// enough"): 0 on the background, the number of steps to the nearest background pixel on
/// the object. Steps go to the 4 neighbours (city-block) or, if `chessboard`, to all 8.
/// Only background inside the image counts; an object without any background stays at `FAR`.
///
/// Hint: start with `FAR` on the object and 0 on the background. In the forward pass
/// (rows top to bottom, each row left to right) set d = min(d, d(left) + 1, d(up) + 1),
/// skipping neighbours outside the image; the backward pass goes bottom-right to top-left
/// with the right and lower neighbours. For the chessboard add the diagonal neighbours
/// already visited: up-left and up-right forward, down-right and down-left backward.
pub fn distance_transform(a: &Image<bool>, chessboard: bool) -> Image<u32> {
    todo!("C3: distance_transform")
}

/// C4. Erosion by the diamond (city-block) or the square (chessboard) of radius `r` as a
/// threshold of the distance transform: object where DT > r.
pub fn erode_by_distance(a: &Image<bool>, r: u32, chessboard: bool) -> Image<bool> {
    todo!("C4: erode_by_distance")
}
