//! Coding exercises for the unit "Self-distillation and foundation models".
//!
//! Write your code in `src/self_distillation/exercise.rs`. Check it with `cargo test -p selfsup self_distillation::`, and run it
//! with `cargo run -p selfsup -- self-distillation` (pictures in `out/self-distillation/`).
//! `src/self_distillation/solution.rs` is a model solution: `cargo test -p selfsup --features solution self_distillation::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p selfsup -- self-distillation`.
pub mod run;
