//! Coding exercises for the unit "Information, entropy and entropy codes".
//!
//! Write your code in `src/entropy/exercise.rs`. Check it with `cargo test -p compression entropy::`, and run it on
//! the micrograph with `cargo run -p compression -- entropy` (pictures in `out/entropy/`).
//! `src/entropy/solution.rs` is a model solution: `cargo test -p compression --features solution entropy::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p compression -- entropy`.
pub mod run;
