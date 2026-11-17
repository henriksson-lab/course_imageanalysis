//! Coding exercises for the unit "Networks and classical pipelines compared".
//!
//! Write your code in `src/cellnets_compare/exercise.rs`. Check it with `cargo test -p cellnets cellnets_compare::`, and run it
//! with `cargo run -p cellnets -- cellnets-compare` (pictures in `out/cellnets-compare/`).
//! `src/cellnets_compare/solution.rs` is a model solution: `cargo test -p cellnets --features solution cellnets_compare::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p cellnets -- cellnets-compare`.
pub mod run;
