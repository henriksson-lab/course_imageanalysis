//! Coding exercises for the unit "Transfer learning".
//!
//! Write your code in `src/transfer/exercise.rs`. Check it with `cargo test -p finetuning transfer::`, and run it
//! with `cargo run -p finetuning -- transfer` (pictures in `out/transfer/`).
//! `src/transfer/solution.rs` is a model solution: `cargo test -p finetuning --features solution transfer::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p finetuning -- transfer`.
pub mod run;
