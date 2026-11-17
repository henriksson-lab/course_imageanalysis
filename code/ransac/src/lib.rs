//! Coding exercises for the unit "Hypothesise and verify: RANSAC".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p ransac`, and run it
//! on random point sets with `cargo run -p ransac` (pictures in `out/ransac/`).
//! `src/solution.rs` is a model solution: `cargo test -p ransac --features solution`.
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
