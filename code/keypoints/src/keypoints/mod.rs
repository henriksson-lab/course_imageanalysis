//! Coding exercises for the unit "Keypoints: corners and blobs across scale".
//!
//! Write your code in `src/keypoints/exercise.rs`. Check it with `cargo test -p keypoints keypoints::`, and run it on
//! the micrograph with `cargo run -p keypoints -- keypoints` (pictures in `out/keypoints/`).
//! `src/keypoints/solution.rs` is a model solution: `cargo test -p keypoints --features solution keypoints::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p keypoints -- keypoints`.
pub mod run;
