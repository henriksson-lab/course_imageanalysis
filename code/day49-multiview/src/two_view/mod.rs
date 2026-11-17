//! Coding exercises for the unit "Two views: matches, epipolar geometry and triangulation".
//!
//! Write your code in `src/two_view/exercise.rs`. Check it with `cargo test -p day49 two_view::`, and run it on
//! a synthetic two-camera scene with `cargo run -p day49 -- two-view` (pictures in `out/two-view/`).
//! `src/two_view/solution.rs` is a model solution: `cargo test -p day49 --features solution two_view::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day49 -- two-view`.
pub mod run;
