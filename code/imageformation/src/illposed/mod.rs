//! Coding exercises for the unit "Running the model backwards: an ill-posed problem".
//!
//! Write your code in `src/illposed/exercise.rs`. Check it with `cargo test -p imageformation illposed::`, and run it on
//! the micrograph with `cargo run -p imageformation -- illposed` (pictures in `out/illposed/`).
//! `src/illposed/solution.rs` is a model solution: `cargo test -p imageformation --features solution illposed::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p imageformation -- illposed`.
pub mod run;
