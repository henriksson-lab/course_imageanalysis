//! Coding exercises for the unit "Training GANs and what goes wrong".
//!
//! Write your code in `src/gan_training/exercise.rs`. Check it with `cargo test -p gans gan_training::`, and run it
//! with `cargo run -p gans -- gan-training` (pictures in `out/gan-training/`).
//! `src/gan_training/solution.rs` is a model solution: `cargo test -p gans --features solution gan_training::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p gans -- gan-training`.
pub mod run;
