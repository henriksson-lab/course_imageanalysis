//! Coding exercises for the unit "Tracking: from detections to tracks".
//!
//! Write your code in `src/track_linking/exercise.rs`. Check it with `cargo test -p day26 track_linking::`, and run the
//! examples with `cargo run -p day26 -- track-linking` (pictures in `out/track-linking/`).
//! `src/track_linking/solution.rs` is a model solution: `cargo test -p day26 --features solution track_linking::`.

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

/// The program of this unit: `cargo run -p day26 -- track-linking`.
pub mod run;
