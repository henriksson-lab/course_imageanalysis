//! Coding exercises for the unit "Learning: loss, gradient descent and backpropagation".
//!
//! Write your code in `src/backprop/exercise.rs`. Check it with `cargo test -p neuralnets backprop::`, and run it
//! with `cargo run -p neuralnets -- backprop` (pictures in `out/backprop/`).
//! `src/backprop/solution.rs` is a model solution: `cargo test -p neuralnets --features solution backprop::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p neuralnets -- backprop`.
pub mod run;
