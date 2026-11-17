//! Coding exercises for the unit "From Dijkstra to fast marching: continuous minimal paths".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p fastmarching`, and run it with
//! `cargo run -p fastmarching` (pictures in `out/fastmarching/`).
//! `src/solution.rs` is a model solution: `cargo test -p fastmarching --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

// A priority queue of pixels for Dijkstra's algorithm (plumbing, filled in).
mod queue;
pub use queue::MinQueue;
