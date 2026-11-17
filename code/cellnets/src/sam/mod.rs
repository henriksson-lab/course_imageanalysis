//! Coding exercises for the unit "Promptable segmentation: SAM".
//!
//! Write your code in `src/sam/exercise.rs`. Check it with `cargo test -p cellnets sam::`, and run it
//! with `cargo run -p cellnets -- sam` (pictures in `out/sam/`).
//! `src/sam/solution.rs` is a model solution: `cargo test -p cellnets --features solution sam::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p cellnets -- sam`.
pub mod run;
