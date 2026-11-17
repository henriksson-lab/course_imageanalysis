//! Coding exercises for the unit "Hough transform: lines".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p hough-lines`, and run it
//! on a synthetic scene with `cargo run -p hough-lines` (pictures in `out/hough-lines/`).
//! `src/solution.rs` is a model solution: `cargo test -p hough-lines --features solution`.
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
