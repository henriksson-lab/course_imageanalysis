//! Coding exercises for the unit "The 2D wavelet transform and sparsity of images".
//!
//! Write your code in `src/dwt2d/exercise.rs`. Check it with `cargo test -p wavelets dwt2d::`, and run it
//! with `cargo run -p wavelets -- dwt2d` (pictures in `out/dwt2d/`).
//! `src/dwt2d/solution.rs` is a model solution: `cargo test -p wavelets --features solution dwt2d::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p wavelets -- dwt2d`.
pub mod run;
