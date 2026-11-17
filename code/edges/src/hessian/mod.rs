//! Coding exercises for the unit "The Hessian".
//!
//! Write your code in `src/hessian/exercise.rs`. Check it with `cargo test -p edges hessian::`, and run it on
//! the test images with `cargo run -p edges -- hessian` (pictures in `out/hessian/`).
//! `src/hessian/solution.rs` is a model solution: `cargo test -p edges --features solution hessian::`.
//! The Gaussian blur of the edges unit is `imgcourse::filters::gaussian_blur`, ready to use.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p edges -- hessian`.
pub mod run;
