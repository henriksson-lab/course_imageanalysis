// Coding exercises for "The DP principle: minimal paths on a grid". Replace each `todo!()`
// with your code, then check it with `cargo test -p dp-paths` and look at `cargo run -p dp-paths`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// C1. The DP table of the cheapest monotone path through a cost image: one pixel per row,
/// each step one row down and at most one column sideways (↙ ↓ ↘). Returns the table `M`,
/// where `M(x, y)` is the cost of the cheapest path from the top row to pixel `(x, y)`, and
/// the pointers: the step `d` (−1, 0 or 1) such that the minimum came from `(x + d, y − 1)`.
/// On a tie, straight wins, then left, then right.
///
/// Hint: fill row 0 with the costs, then the rows 1, 2, ... in order. Try the steps in the
/// order `[0, -1, 1]` and replace the best only if a candidate is strictly smaller: that is
/// the tie rule. Skip candidates outside the image.
pub fn dp_table(cost: &Image<f32>) -> (Image<f32>, Image<isize>) {
    todo!("C1: dp_table")
}

/// C2. Backtracking: start at the smallest `M` of the last row (the leftmost one on a tie)
/// and follow the pointers up. Returns the column of the path in every row, from the top
/// row to the bottom row.
pub fn backtrack(m: &Image<f32>, pointer: &Image<isize>) -> Vec<usize> {
    todo!("C2: backtrack")
}

/// C3. The greedy path: start at the cheapest pixel of the top row (the leftmost one on a
/// tie), then always step to the cheapest of the (up to) three pixels below (on a tie:
/// straight, then left, then right). Returns the column in every row.
pub fn greedy_path(cost: &Image<f32>) -> Vec<usize> {
    todo!("C3: greedy_path")
}

/// The cost of a path given by its column in every row: the sum of its pixels.
pub fn path_cost(cost: &Image<f32>, columns: &[usize]) -> f32 {
    columns.iter().enumerate().map(|(y, &x)| cost[(x, y)]).sum()
}

/// C4. The number of monotone paths from the top row to every pixel of a `width × height`
/// image: the same table as C1, with "sum" in place of "cost + min". (In `f64`, because the
/// numbers grow very fast.)
pub fn count_paths(width: usize, height: usize) -> Image<f64> {
    todo!("C4: count_paths")
}

/// C5a. The forward pass of the city-block distance transform: visit the pixels row by row,
/// left to right, and set `D(x, y) = min(D(x, y), D(x, y − 1) + 1, D(x − 1, y) + 1)`.
/// `D` starts at 0 on the background pixels (`true`) and ∞ elsewhere.
///
/// Hint: `background.map(|b| if b { 0.0 } else { f32::INFINITY })` is the start.
pub fn distance_forward(background: &Image<bool>) -> Image<f32> {
    todo!("C5a: distance_forward")
}

/// C5b. The full city-block distance transform: the forward pass, then a backward pass in
/// the opposite order (bottom to top, right to left) that looks down and right.
pub fn distance_two_passes(background: &Image<bool>) -> Image<f32> {
    todo!("C5b: distance_two_passes")
}
