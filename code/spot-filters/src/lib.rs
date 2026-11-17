//! Coding exercises for the unit "Finding spots: filter, then local maxima".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p spot-filters`, and run it
//! on a synthetic spot image with `cargo run -p spot-filters` (pictures in `out/spot-filters/`).
//! `src/solution.rs` is a model solution: `cargo test -p spot-filters --features solution`.
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
