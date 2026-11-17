//! Coding exercises for the unit "When there is no order: Dijkstra and live-wire".
//!
//! Write your code in `src/livewire/exercise.rs`. Check it with `cargo test -p dynprog livewire::`, and run it with
//! `cargo run -p dynprog -- livewire` (pictures in `out/livewire/`).
//! `src/livewire/solution.rs` is a model solution: `cargo test -p dynprog --features solution livewire::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

// A priority queue of pixels for Dijkstra's algorithm (plumbing, filled in, in `src/common.rs`).
pub use crate::common::MinQueue;

/// The program of this unit: `cargo run -p dynprog -- livewire`.
pub mod run;
