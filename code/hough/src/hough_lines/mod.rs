//! Coding exercises for the unit "Hough transform: lines".
//!
//! Write your code in `src/hough_lines/exercise.rs`. Check it with `cargo test -p hough hough_lines::`, and run it
//! on a synthetic scene with `cargo run -p hough -- hough-lines` (pictures in `out/hough-lines/`).
//! `src/hough_lines/solution.rs` is a model solution: `cargo test -p hough --features solution hough_lines::`.
//! `src/common.rs` makes the test images and draws the results; it is not part of the exercise.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p hough -- hough-lines`.
pub mod run;
