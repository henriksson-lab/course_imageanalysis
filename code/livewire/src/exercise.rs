// Coding exercises for "When there is no order: Dijkstra and live-wire". Replace each `todo!()`
// with your code, then check it with `cargo test -p livewire` and look at `cargo run -p livewire`.
#![allow(unused_variables, unused_imports)]

use crate::MinQueue;
use imgcourse::{Border, Image};

/// The result of Dijkstra's algorithm: the shortest-path tree.
pub struct ShortestPaths {
    /// The cost of the cheapest path from a seed to every pixel (∞ if not reached).
    pub distance: Image<f32>,
    /// The neighbour each pixel's cheapest path comes from (`None` for the seeds).
    pub predecessor: Image<Option<(usize, usize)>>,
    /// The pixels in the order in which they became final.
    pub order: Vec<(usize, usize)>,
}

/// C1. Dijkstra's algorithm on the pixel grid. A path pays the cost of every pixel it enters,
/// times the step length (1, or √2 for a diagonal step when `eight` is true); a seed starts
/// with its own cost, `d(s) = C(s)`. Repeatedly make the unfinished pixel with the smallest
/// tentative distance final, and update its neighbours.
///
/// Hint: keep three images: the distance (start: `f32::INFINITY`), the predecessor (start:
/// `None`) and "final" (start: `false`). Push the seeds into a `MinQueue`. Then, while
/// `queue.pop()` gives a pixel: skip it if it is already final (an old entry), else make it
/// final, add it to `order`, and for every neighbour inside the image that is not final, if
/// `d + length · C(neighbour)` is smaller than its distance, lower it, set its predecessor and
/// push it.
pub fn dijkstra(cost: &Image<f32>, seeds: &[(usize, usize)], eight: bool) -> ShortestPaths {
    todo!("C1: dijkstra")
}

/// C2. The cheapest path from a seed to `target`: follow the predecessors back to a seed.
/// Returns the pixels from the seed to the target.
pub fn backtrack(paths: &ShortestPaths, target: (usize, usize)) -> Vec<(usize, usize)> {
    todo!("C2: backtrack")
}

/// C3. One raster pass instead of Dijkstra, from the seed at the top-left pixel: visit the
/// pixels row by row, left to right, with `d(x, y) = C(x, y) + min(d(x, y − 1), d(x − 1, y))`
/// and `d(0, 0) = C(0, 0)`. Exact only where the cheapest path never goes up or left.
pub fn raster_pass(cost: &Image<f32>) -> Image<f32> {
    todo!("C3: raster_pass")
}

/// A small constant cost per step, so that flat regions do not give free detours.
pub const EPSILON: f32 = 0.02;

/// C4. The live-wire cost of entering a pixel, from the gradient magnitude `G` (central
/// differences, the edge pixel copied at the border): `EPSILON + 1 − G / G_max`. Low on
/// strong edges, about 1 in flat regions.
///
/// Hint: first the gradient magnitude image (as in the filtering unit), then `g.max()`.
pub fn edge_cost(img: &Image<f32>) -> Image<f32> {
    todo!("C4: edge_cost")
}

