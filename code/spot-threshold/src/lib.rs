//! Coding exercises for the unit "How many false spots? Thresholds from noise statistics".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p spot-threshold`, and run it
//! on images of noise and spots with `cargo run -p spot-threshold` (pictures in `out/spot-threshold/`).
//! `src/solution.rs` is a model solution: `cargo test -p spot-threshold --features solution`.
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
