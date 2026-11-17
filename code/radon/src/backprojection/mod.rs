//! Coding exercises for the unit "Back-projection and the Fourier slice theorem".
//!
//! Write your code in `src/backprojection/exercise.rs`. Check it with `cargo test -p radon backprojection::`, and run it on
//! test images with `cargo run -p radon -- backprojection` (pictures in `out/backprojection/`).
//! `src/backprojection/solution.rs` is a model solution: `cargo test -p radon --features solution backprojection::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p radon -- backprojection`.
pub mod run;
