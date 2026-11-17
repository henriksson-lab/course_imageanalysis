//! Coding exercises for the unit "Few labels: annotation and the human in the loop".
//!
//! Write your code in `src/few_labels/exercise.rs`. Check it with `cargo test -p finetuning few_labels::`, and run it
//! with `cargo run -p finetuning -- few-labels` (pictures in `out/few-labels/`).
//! `src/few_labels/solution.rs` is a model solution: `cargo test -p finetuning --features solution few_labels::`.

/// Given code: the pool of cells and the classifier.
pub mod model;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p finetuning -- few-labels`.
pub mod run;
