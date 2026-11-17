//! Coding exercises for the unit "Tracking: from detections to tracks".
//!
//! Write your code in `src/track_linking/exercise.rs`. Check it with `cargo test -p tracking track_linking::`, and run the
//! examples with `cargo run -p tracking -- track-linking` (pictures in `out/track-linking/`).
//! `src/track_linking/solution.rs` is a model solution: `cargo test -p tracking --features solution track_linking::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

// Provided: a fast exact assignment solver (not an exercise), in `src/common.rs`.
pub use crate::common::hungarian;

/// The program of this unit: `cargo run -p tracking -- track-linking`.
pub mod run;
