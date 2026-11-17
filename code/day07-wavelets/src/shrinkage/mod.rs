//! Coding exercises for the unit "Denoising by wavelet shrinkage".
//!
//! Write your code in `src/shrinkage/exercise.rs`. Check it with `cargo test -p day07 shrinkage::`, and run it
//! with `cargo run -p day07 -- shrinkage` (pictures in `out/shrinkage/`).
//! `src/shrinkage/solution.rs` is a model solution: `cargo test -p day07 --features solution shrinkage::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day07 -- shrinkage`.
pub mod run;
