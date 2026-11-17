//! Coding exercises for the unit "Tracking: gaps, divisions and lineage trees".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p track-events`, and run the
//! examples with `cargo run -p track-events` (pictures in `out/track-events/`).
//! `src/solution.rs` is a model solution: `cargo test -p track-events --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

// Provided: global assignment with alternative costs (not an exercise; see track-linking).
mod assign;
pub use assign::{hungarian, join_with_alternatives};
