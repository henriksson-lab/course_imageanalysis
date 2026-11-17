//! Coding exercises for the unit "Neural networks: matrix, bend, matrix".
//!
//! Write your code in `src/neuralnet/exercise.rs`. Check it with `cargo test -p neuralnets neuralnet::`, and run it
//! with `cargo run -p neuralnets -- neuralnet` (pictures in `out/neuralnet/`).
//! `src/neuralnet/solution.rs` is a model solution: `cargo test -p neuralnets --features solution neuralnet::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p neuralnets -- neuralnet`.
pub mod run;
