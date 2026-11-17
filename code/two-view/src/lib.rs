//! Coding exercises for the unit "Two views: matches, epipolar geometry and triangulation".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p two-view`, and run it on
//! a synthetic two-camera scene with `cargo run -p two-view` (pictures in `out/two-view/`).
//! `src/solution.rs` is a model solution: `cargo test -p two-view --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
