// Coding exercises for "Grey-level morphology: top-hat and gradient". Replace each `todo!()`
// with your code, then check it with `cargo test -p day13 morph_grey::` and look at `cargo run -p day13 -- morph-grey`.
#![allow(unused_variables, unused_imports)]
//
// A flat structuring element (SE) is a list of offsets (dx, dy) from its origin, no
// weights. At the image border only the pixels inside the image count (as in the exercises).

use imgcourse::Image;

/// A digital disc of radius `r`: all offsets with dx² + dy² ≤ r² (given, in `src/common.rs`).
pub use crate::common::disc;

/// A horizontal window of `width` samples centred on the origin (`width` odd), for 1D
/// signals stored as images of height 1.
pub fn window(width: usize) -> Vec<(isize, isize)> {
    let k = (width / 2) as isize;
    (-k..=k).map(|dx| (dx, 0)).collect()
}

/// The values of `img` at x + sign · b for all offsets b of the SE that land inside the image
/// (`sign` 1 for erosion, −1 for dilation).
pub fn values_at(img: &Image<f32>, x: usize, y: usize, se: &[(isize, isize)], sign: isize) -> Vec<f32> {
    let mut v = Vec::with_capacity(se.len());
    for &(dx, dy) in se {
        let (u, w) = (x as isize + sign * dx, y as isize + sign * dy);
        if img.contains(u, w) {
            v.push(img[(u as usize, w as usize)]);
        }
    }
    v
}

/// C1a. Erosion with a flat SE: the minimum of I(x + b) over the SE (a min filter).
///
/// Hint: `values_at(img, x, y, se, 1)` gives the values under the SE (inside the image);
/// fold them with `f32::min`, starting from `f32::INFINITY`.
pub fn erode(img: &Image<f32>, se: &[(isize, isize)]) -> Image<f32> {
    todo!("C1a: erode")
}

/// C1b. Dilation with a flat SE: the maximum of I(x − b) over the SE (a max filter).
///
/// Hint: the same with `sign = -1` (x − b) and `f32::max`.
pub fn dilate(img: &Image<f32>, se: &[(isize, isize)]) -> Image<f32> {
    todo!("C1b: dilate")
}

/// C2a. Opening: erode, then dilate.
pub fn open(img: &Image<f32>, se: &[(isize, isize)]) -> Image<f32> {
    todo!("C2a: open")
}

/// C2b. Closing: dilate, then erode.
pub fn close(img: &Image<f32>, se: &[(isize, isize)]) -> Image<f32> {
    todo!("C2b: close")
}

/// C2c. White top-hat: the image minus its opening (bright details smaller than the SE).
///
/// Hint: `zip_map` subtracts two images pixel by pixel.
pub fn white_tophat(img: &Image<f32>, se: &[(isize, isize)]) -> Image<f32> {
    todo!("C2c: white_tophat")
}

/// C2d. Black top-hat: the closing minus the image (dark details smaller than the SE).
pub fn black_tophat(img: &Image<f32>, se: &[(isize, isize)]) -> Image<f32> {
    todo!("C2d: black_tophat")
}

/// C3. The morphological gradient: dilation minus erosion (the range in the window).
pub fn gradient(img: &Image<f32>, se: &[(isize, isize)]) -> Image<f32> {
    todo!("C3: gradient")
}

/// C4. The median filter: sort the values I(x + b) of the window and take the middle one
/// (for an even number of values, the upper of the two middle ones).
///
/// Hint: sort with `v.sort_by(|a, b| a.total_cmp(b))` (floats have no plain `sort`).
pub fn median(img: &Image<f32>, se: &[(isize, isize)]) -> Image<f32> {
    todo!("C4: median")
}
