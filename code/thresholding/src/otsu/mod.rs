//! Coding exercises for the unit "Choosing the threshold from the histogram".
//!
//! Write your code in `src/otsu/exercise.rs`. Check it with `cargo test -p thresholding otsu::`, and run it on
//! a test image and the micrograph with `cargo run -p thresholding -- otsu` (pictures in `out/otsu/`).
//! `src/otsu/solution.rs` is a model solution: `cargo test -p thresholding --features solution otsu::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p thresholding -- otsu`.
pub mod run;
