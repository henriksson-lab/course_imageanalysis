//! Coding exercises for the unit "Filtered back-projection and sampling".
//!
//! Write your code in `src/fbp/exercise.rs`. Check it with `cargo test -p radon fbp::`, and run it on
//! test images with `cargo run -p radon -- fbp` (pictures in `out/fbp/`).
//! `src/fbp/solution.rs` is a model solution: `cargo test -p radon --features solution fbp::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p radon -- fbp`.
pub mod run;
