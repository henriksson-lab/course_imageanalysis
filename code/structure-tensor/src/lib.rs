//! Coding exercises for the unit "The structure tensor".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p structure-tensor`, and run it on
//! the test images with `cargo run -p structure-tensor` (pictures in `out/structure-tensor/`).
//! `src/solution.rs` is a model solution: `cargo test -p structure-tensor --features solution`.
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
