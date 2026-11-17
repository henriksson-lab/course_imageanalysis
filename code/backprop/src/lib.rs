//! Coding exercises for the unit "Learning: loss, gradient descent and backpropagation".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p backprop`, and run it
//! with `cargo run -p backprop` (pictures in `out/backprop/`).
//! `src/solution.rs` is a model solution: `cargo test -p backprop --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
