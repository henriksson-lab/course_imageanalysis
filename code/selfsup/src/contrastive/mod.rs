//! Coding exercises for the unit "Contrastive learning".
//!
//! Write your code in `src/contrastive/exercise.rs`. Check it with `cargo test -p selfsup contrastive::`, and run it
//! with `cargo run -p selfsup -- contrastive` (pictures in `out/contrastive/`).
//! `src/contrastive/solution.rs` is a model solution: `cargo test -p selfsup --features solution contrastive::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p selfsup -- contrastive`.
pub mod run;
