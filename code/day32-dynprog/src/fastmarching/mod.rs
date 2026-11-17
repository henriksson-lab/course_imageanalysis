//! Coding exercises for the unit "From Dijkstra to fast marching: continuous minimal paths".
//!
//! Write your code in `src/fastmarching/exercise.rs`. Check it with `cargo test -p day32 fastmarching::`, and run it with
//! `cargo run -p day32 -- fastmarching` (pictures in `out/fastmarching/`).
//! `src/fastmarching/solution.rs` is a model solution: `cargo test -p day32 --features solution fastmarching::`.

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

/// The program of this unit: `cargo run -p day32 -- fastmarching`.
pub mod run;
