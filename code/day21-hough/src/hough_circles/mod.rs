//! Coding exercises for the unit "Hough transform: circles".
//!
//! Write your code in `src/hough_circles/exercise.rs`. Check it with `cargo test -p day21 hough_circles::`, and run it
//! on synthetic discs and nuclei with `cargo run -p day21 -- hough-circles` (pictures in `out/hough-circles/`).
//! `src/hough_circles/solution.rs` is a model solution: `cargo test -p day21 --features solution hough_circles::`.
//! `src/common.rs` and `src/hough_circles/sim.rs` make the test images and draw the results; they are not part of the exercise.

pub mod sim;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day21 -- hough-circles`.
pub mod run;
