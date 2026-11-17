//! Coding exercises for the unit "The Hessian".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p hessian`, and run it on
//! the test images with `cargo run -p hessian` (pictures in `out/hessian/`).
//! `src/solution.rs` is a model solution: `cargo test -p hessian --features solution`.
//! `src/given.rs` has the Gaussian blur of the edges unit, ready to use.

pub mod given;
pub use given::*;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
