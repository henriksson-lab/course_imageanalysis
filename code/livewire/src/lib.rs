//! Coding exercises for the unit "When there is no order: Dijkstra and live-wire".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p livewire`, and run it with
//! `cargo run -p livewire` (pictures in `out/livewire/`).
//! `src/solution.rs` is a model solution: `cargo test -p livewire --features solution`.

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
