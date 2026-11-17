//! Coding exercises for the unit "Sub-pixel localisation".
//!
//! Write your code in `src/spot_fit/exercise.rs`. Check it with `cargo test -p pointdetection spot_fit::`, and run it
//! on simulated spots with `cargo run -p pointdetection -- spot-fit` (pictures in `out/spot-fit/`).
//! `src/spot_fit/solution.rs` is a model solution: `cargo test -p pointdetection --features solution spot_fit::`.
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

/// The program of this unit: `cargo run -p pointdetection -- spot-fit`.
pub mod run;
