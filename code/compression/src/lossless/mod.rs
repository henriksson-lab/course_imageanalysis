//! Coding exercises for the unit "Prediction, dictionaries and lossless formats".
//!
//! Write your code in `src/lossless/exercise.rs`. Check it with `cargo test -p compression lossless::`, and run it on
//! the micrograph with `cargo run -p compression -- lossless` (pictures in `out/lossless/`).
//! `src/lossless/solution.rs` is a model solution: `cargo test -p compression --features solution lossless::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p compression -- lossless`.
pub mod run;
