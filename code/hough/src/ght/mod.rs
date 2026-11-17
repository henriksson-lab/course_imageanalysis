//! Coding exercises for the unit "The generalised Hough transform".
//!
//! Write your code in `src/ght/exercise.rs`. Check it with `cargo test -p hough ght::`, and run it
//! on a synthetic scene of keys with `cargo run -p hough -- ght` (pictures in `out/ght/`).
//! `src/ght/solution.rs` is a model solution: `cargo test -p hough --features solution ght::`.
//! `src/common.rs` and `src/ght/sim.rs` make the test images and draw the results; they are not part of the exercise.

pub mod sim;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p hough -- ght`.
pub mod run;
