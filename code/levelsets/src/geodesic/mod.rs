//! Coding exercises for the unit "Edge-based segmentation: geodesic active contours".
//!
//! Write your code in `src/geodesic/exercise.rs`. Check it with `cargo test -p levelsets geodesic::`, and run it
//! with `cargo run -p levelsets -- geodesic` (pictures in `out/geodesic/`).
//! `src/geodesic/solution.rs` is a model solution: `cargo test -p levelsets --features solution geodesic::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p levelsets -- geodesic`.
pub mod run;
