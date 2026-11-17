// Coding exercises for "Stereo: from disparity to depth".
// Replace each `todo!()` with your code, then check it with `cargo test -p stereo`
// and look at `cargo run -p stereo`.
// A rectified pair: the left pixel (x, y) matches the right pixel (x − d, y), d ≥ 0 the
// disparity. A cost volume is one image of costs per disparity: `costs[d][(x, y)]`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};

/// C1. The SAD cost volume: for every disparity d in 0..=max_d and every left pixel (x, y),
/// the sum over the (2 radius + 1)² window of |L(x + i, y + j) − R(x + i − d, y + j)|. Pixels
/// outside the images are read as 0.
///
/// Hint: `left.get_border(x + i, y + j, Border::Zero)` reads outside the image too; work
/// with `isize` for x, y, i, j and d.
pub fn sad_cost_volume(left: &Image<f32>, right: &Image<f32>, max_d: usize, radius: usize) -> Vec<Image<f32>> {
    todo!("C1: sad_cost_volume")
}

/// C2a. The census transform with a 3 × 3 window: for every pixel an 8-bit code, one bit per
/// neighbour, row by row and skipping the centre, the first neighbour in the highest bit.
/// A bit is 1 if the neighbour is darker than the centre. Outside the image, the edge is copied.
///
/// Hint: `img.map_xy(|x, y, centre| ...)`; build the code with `code = (code << 1) | bit`.
pub fn census(img: &Image<f32>) -> Image<u32> {
    todo!("C2a: census")
}

/// C2b. The Hamming distance of two codes: the number of bits in which they differ.
///
/// Hint: `a ^ b` has a 1 where the bits differ; count them (or use `count_ones`).
pub fn hamming(a: u32, b: u32) -> u32 {
    todo!("C2b: hamming")
}

/// C3. Winner-take-all: at every pixel, the disparity with the lowest cost (the smallest d on
/// a tie).
pub fn winner_take_all(costs: &[Image<f32>]) -> Image<f32> {
    todo!("C3: winner_take_all")
}

/// C4. Scanline dynamic programming on one row: `costs[x][d]` for the pixels x of the row.
/// Find the disparity profile with the smallest energy: the sum of its costs, plus `p1` for
/// every step of 1 between neighbours and `p2` for every larger jump (slide "Scanline dynamic
/// programming"). Fill a table of cumulative costs column by column with a pointer to the best
/// predecessor, then backtrack from the best end.
///
/// Hint: `total[x][d] = costs[x][d] + min over e of (total[x − 1][e] + jump(e, d))`, and
/// `from[x][d]` = the best e. Backtrack with `path[x − 1] = from[x][path[x]]`.
pub fn scanline_dp(costs: &[Vec<f32>], p1: f32, p2: f32) -> Vec<usize> {
    todo!("C4: scanline_dp")
}

/// C5. Sub-pixel disparity: the offset of the minimum of the parabola through the costs at
/// d − 1, d and d + 1 (slide "The math: the sub-pixel parabola"), between −½ and ½.
///
/// Hint: return 0 if the curvature C₋ − 2C₀ + C₊ is not positive.
pub fn subpixel_offset(c_minus: f32, c0: f32, c_plus: f32) -> f32 {
    todo!("C5: subpixel_offset")
}
