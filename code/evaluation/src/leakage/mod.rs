//! Coding exercises for the unit "Honest validation: leakage, splits and domain shift".
//!
//! Write your code in `src/leakage/exercise.rs`. Check it with `cargo test -p evaluation leakage::`, and run it
//! with `cargo run -p evaluation -- leakage` (pictures in `out/leakage/`).
//! `src/leakage/solution.rs` is a model solution: `cargo test -p evaluation --features solution leakage::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p evaluation -- leakage`.
pub mod run;
