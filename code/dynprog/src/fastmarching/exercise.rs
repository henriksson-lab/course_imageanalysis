// Coding exercises for "From Dijkstra to fast marching: continuous minimal paths". Replace each
// `todo!()` with your code, then check it with `cargo test -p dynprog fastmarching::` and look at
// `cargo run -p dynprog -- fastmarching`.
#![allow(unused_variables, unused_imports)]

use crate::fastmarching::MinQueue;
use imgcourse::Image;

/// C1. The upwind update of fast marching: the arrival time at a pixel from `a`, the smaller
/// known value of its left and right neighbours, and `b`, the smaller known value of its
/// upper and lower neighbours (`f32::INFINITY` if none is known), with cost `p` per unit
/// length. If |a − b| < p: the larger root of (T − a)² + (T − b)² = p²; otherwise
/// min(a, b) + p (Dijkstra's update).
pub fn upwind_update(a: f32, b: f32, p: f32) -> f32 {
    todo!("C1: upwind_update")
}

/// C2. Fast marching: the arrival time `T` of a front that starts at the seeds (`T = 0`)
/// and crosses pixel `(x, y)` at cost `cost(x, y)` per unit length. Dijkstra's order (the
/// unfinished pixel with the smallest tentative `T` is final), with the upwind update (C1)
/// from the final 4-neighbours.
///
/// Hint: the same loop as Dijkstra (livewire unit), with a "final" image. When a pixel becomes
/// final, recompute each of its 4 neighbours that is not final yet: `a` from its left and right
/// neighbours, `b` from its upper and lower neighbours, counting only final ones (∞ otherwise),
/// then C1. If the result is smaller than the neighbour's value, store it and push it.
pub fn fast_marching(cost: &Image<f32>, seeds: &[(usize, usize)]) -> Image<f32> {
    todo!("C2: fast_marching")
}

/// C3. For comparison, Dijkstra with the same cost: a step into pixel `v` costs
/// `length · cost(v)`, with length 1 (or √2 for a diagonal step when `eight` is true); the
/// seeds start at 0.
pub fn dijkstra_time(cost: &Image<f32>, seeds: &[(usize, usize)], eight: bool) -> Image<f32> {
    todo!("C3: dijkstra_time")
}

/// C4. The minimal path from `target` back to a seed, by gradient descent on `T`: repeatedly
/// take a step of length `step` against the gradient of `T` (central differences at the
/// nearest pixel; a one-sided difference where a neighbour is outside or unreached). Stop at
/// a pixel with `T = 0` (a seed), or after 100 000 steps. Returns the positions visited,
/// from the target to the seed.
///
/// Hint: `(x.round() as isize, y.round() as isize)` is the nearest pixel. Divide the gradient
/// by its length, so that every step has the same length.
pub fn descend(t: &Image<f32>, target: (f32, f32), step: f32) -> Vec<(f32, f32)> {
    todo!("C4: descend")
}
