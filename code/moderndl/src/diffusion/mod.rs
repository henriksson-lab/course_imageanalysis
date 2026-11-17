//! Coding exercises for the unit "Diffusion models and generated detail".
//!
//! Write your code in `src/diffusion/exercise.rs`. Check it with `cargo test -p moderndl diffusion::`, and run it
//! with `cargo run -p moderndl -- diffusion` (pictures in `out/diffusion/`).
//! `src/diffusion/solution.rs` is a model solution: `cargo test -p moderndl --features solution diffusion::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p moderndl -- diffusion`.
pub mod run;
