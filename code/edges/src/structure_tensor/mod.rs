//! Coding exercises for the unit "The structure tensor".
//!
//! Write your code in `src/structure_tensor/exercise.rs`. Check it with `cargo test -p edges structure_tensor::`, and run it on
//! the test images with `cargo run -p edges -- structure-tensor` (pictures in `out/structure-tensor/`).
//! `src/structure_tensor/solution.rs` is a model solution: `cargo test -p edges --features solution structure_tensor::`.
//! `src/given.rs` has the Gaussian blur and the gradient of the edges unit, ready to use.

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

/// The program of this unit: `cargo run -p edges -- structure-tensor`.
pub mod run;
