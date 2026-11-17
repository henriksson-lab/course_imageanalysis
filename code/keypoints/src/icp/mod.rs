//! Coding exercises for the unit "Aligning point sets".
//!
//! Write your code in `src/icp/exercise.rs`. Check it with `cargo test -p keypoints icp::`, and run it on
//! synthetic nuclei with `cargo run -p keypoints -- icp` (pictures in `out/icp/`).
//! `src/icp/solution.rs` is a model solution: `cargo test -p keypoints --features solution icp::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p keypoints -- icp`.
pub mod run;
