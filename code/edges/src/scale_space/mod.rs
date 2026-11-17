//! Coding exercises for the unit "Scale space, blobs and pyramids".
//!
//! Write your code in `src/scale_space/exercise.rs`. Check it with `cargo test -p edges scale_space::`, and run it
//! on the test images with `cargo run -p edges -- scale-space` (pictures in `out/scale-space/`).
//! `src/scale_space/solution.rs` is a model solution: `cargo test -p edges --features solution scale_space::`.
//! The Gaussian blur of the edges unit is `imgcourse::filters::gaussian_blur`, ready to use.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p edges -- scale-space`.
pub mod run;
