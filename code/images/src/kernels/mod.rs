//! Coding exercises for the unit "Filtering & convolution".
//!
//! Write your code in `src/kernels/exercise.rs`. Check it with `cargo test -p images kernels::`, and run it on
//! the micrograph with `cargo run -p images -- kernels` (pictures in `out/kernels/`).
//! `src/kernels/solution.rs` is a model solution: `cargo test -p images --features solution kernels::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p images -- kernels`.
pub mod run;
