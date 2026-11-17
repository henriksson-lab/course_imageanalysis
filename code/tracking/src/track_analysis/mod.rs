//! Coding exercises for the unit "Tracking: speed, MSD and modes of motion".
//!
//! Write your code in `src/track_analysis/exercise.rs`. Check it with `cargo test -p tracking track_analysis::`, and run the
//! examples with `cargo run -p tracking -- track-analysis` (pictures in `out/track-analysis/`).
//! `src/track_analysis/solution.rs` is a model solution: `cargo test -p tracking --features solution track_analysis::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p tracking -- track-analysis`.
pub mod run;
