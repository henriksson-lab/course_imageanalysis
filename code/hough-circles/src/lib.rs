//! Coding exercises for the unit "Hough transform: circles".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p hough-circles`, and run it
//! on synthetic discs and nuclei with `cargo run -p hough-circles` (pictures in `out/hough-circles/`).
//! `src/solution.rs` is a model solution: `cargo test -p hough-circles --features solution`.
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
