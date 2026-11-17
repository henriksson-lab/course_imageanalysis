//! Coding exercises for the unit "The Hessian".
//!
//! Write your code in `src/hessian/exercise.rs`. Check it with `cargo test -p day02 hessian::`, and run it on
//! the test images with `cargo run -p day02 -- hessian` (pictures in `out/hessian/`).
//! `src/hessian/solution.rs` is a model solution: `cargo test -p day02 --features solution hessian::`.
//! The Gaussian blur of the edges unit is `imgcourse::filters::gaussian_blur`, ready to use.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day02 -- hessian`.
pub mod run;
