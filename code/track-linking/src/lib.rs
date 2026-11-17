//! Coding exercises for the unit "Tracking: from detections to tracks".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p track-linking`, and run the
//! examples with `cargo run -p track-linking` (pictures in `out/track-linking/`).
//! `src/solution.rs` is a model solution: `cargo test -p track-linking --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

// Provided: a fast exact assignment solver (not an exercise).
mod assign;
pub use assign::hungarian;
