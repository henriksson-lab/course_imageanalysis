//! Coding exercises for the unit "Sub-pixel localisation".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p spot-fit`, and run it
//! on simulated spots with `cargo run -p spot-fit` (pictures in `out/spot-fit/`).
//! `src/solution.rs` is a model solution: `cargo test -p spot-fit --features solution`.
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
