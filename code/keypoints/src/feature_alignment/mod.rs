//! Coding exercises for the unit "From matches to a transform".
//!
//! Write your code in `src/feature_alignment/exercise.rs`. Check it with `cargo test -p keypoints feature_alignment::`, and run it on
//! the micrograph with `cargo run -p keypoints -- feature-alignment` (pictures in `out/feature-alignment/`).
//! `src/feature_alignment/solution.rs` is a model solution: `cargo test -p keypoints --features solution feature_alignment::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p keypoints -- feature-alignment`.
pub mod run;
