//! Coding exercises for the unit "Rectangle features".
//!
//! Write your code in `src/rect_features/exercise.rs`. Check it with `cargo test -p integral rect_features::`, and run it on
//! the micrograph with `cargo run -p integral -- rect-features` (pictures in `out/rect-features/`).
//! `src/rect_features/solution.rs` is a model solution: `cargo test -p integral --features solution rect_features::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p integral -- rect-features`.
pub mod run;
