//! Coding exercises for the unit "Reading training curves".
//!
//! Write your code in `src/training_curves/exercise.rs`. Check it with `cargo test -p training training_curves::`, and run it
//! with `cargo run -p training -- training-curves` (pictures in `out/training-curves/`).
//! `src/training_curves/solution.rs` is a model solution: `cargo test -p training --features solution training_curves::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p training -- training-curves`.
pub mod run;
