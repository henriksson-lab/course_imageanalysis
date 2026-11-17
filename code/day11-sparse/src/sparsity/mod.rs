//! Coding exercises for the unit "Sparsity and the ℓ1 penalty".
//!
//! Write your code in `src/sparsity/exercise.rs`. Check it with `cargo test -p day11 sparsity::`, and run it on
//! the micrograph with `cargo run -p day11 -- sparsity` (pictures in `out/sparsity/`).
//! `src/sparsity/solution.rs` is a model solution: `cargo test -p day11 --features solution sparsity::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day11 -- sparsity`.
pub mod run;
