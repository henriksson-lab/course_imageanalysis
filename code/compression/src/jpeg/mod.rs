//! Coding exercises for the unit "The JPEG pipeline, artefacts and quality metrics".
//!
//! Write your code in `src/jpeg/exercise.rs`. Check it with `cargo test -p compression jpeg::`, and run it on
//! the micrograph with `cargo run -p compression -- jpeg` (pictures in `out/jpeg/`).
//! `src/jpeg/solution.rs` is a model solution: `cargo test -p compression --features solution jpeg::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p compression -- jpeg`.
pub mod run;
