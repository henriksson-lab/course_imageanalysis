//! Coding exercises for the unit "Moving a curve: the level set equation".
//!
//! Write your code in `src/levelset_eq/exercise.rs`. Check it with `cargo test -p levelsets levelset_eq::`, and run it
//! with `cargo run -p levelsets -- levelset-eq` (pictures in `out/levelset-eq/`).
//! `src/levelset_eq/solution.rs` is a model solution: `cargo test -p levelsets --features solution levelset_eq::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p levelsets -- levelset-eq`.
pub mod run;
