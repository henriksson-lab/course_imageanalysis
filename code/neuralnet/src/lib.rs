//! Coding exercises for the unit "Neural networks: matrix, bend, matrix".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p neuralnet`, and run it
//! with `cargo run -p neuralnet` (pictures in `out/neuralnet/`).
//! `src/solution.rs` is a model solution: `cargo test -p neuralnet --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
