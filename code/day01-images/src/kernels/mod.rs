//! Coding exercises for the unit "Filtering & convolution".
//!
//! Write your code in `src/kernels/exercise.rs`. Check it with `cargo test -p day01 kernels::`, and run it on
//! the micrograph with `cargo run -p day01 -- kernels` (pictures in `out/kernels/`).
//! `src/kernels/solution.rs` is a model solution: `cargo test -p day01 --features solution kernels::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day01 -- kernels`.
pub mod run;
