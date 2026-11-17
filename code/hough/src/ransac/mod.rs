//! Coding exercises for the unit "Hypothesise and verify: RANSAC".
//!
//! Write your code in `src/ransac/exercise.rs`. Check it with `cargo test -p hough ransac::`, and run it
//! on random point sets with `cargo run -p hough -- ransac` (pictures in `out/ransac/`).
//! `src/ransac/solution.rs` is a model solution: `cargo test -p hough --features solution ransac::`.
//! `src/sim.rs` makes the test images and draws the results; it is not part of the exercise.

pub mod sim;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p hough -- ransac`.
pub mod run;
