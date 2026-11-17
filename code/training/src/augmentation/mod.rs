//! Coding exercises for the unit "Regularisation and data augmentation".
//!
//! Write your code in `src/augmentation/exercise.rs`. Check it with `cargo test -p training augmentation::`, and run it
//! with `cargo run -p training -- augmentation` (pictures in `out/augmentation/`).
//! `src/augmentation/solution.rs` is a model solution: `cargo test -p training --features solution augmentation::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p training -- augmentation`.
pub mod run;
