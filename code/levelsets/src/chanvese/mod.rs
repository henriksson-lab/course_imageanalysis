//! Coding exercises for the unit "Region-based segmentation: Chan–Vese".
//!
//! Write your code in `src/chanvese/exercise.rs`. Check it with `cargo test -p levelsets chanvese::`, and run it
//! with `cargo run -p levelsets -- chanvese` (pictures in `out/chanvese/`).
//! `src/chanvese/solution.rs` is a model solution: `cargo test -p levelsets --features solution chanvese::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p levelsets -- chanvese`.
pub mod run;
