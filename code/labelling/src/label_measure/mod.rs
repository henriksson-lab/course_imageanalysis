//! Coding exercises for the unit "Working with label images".
//!
//! Write your code in `src/label_measure/exercise.rs`. Check it with `cargo test -p labelling label_measure::`, and run it on
//! synthetic nuclei with `cargo run -p labelling -- label-measure` (pictures in `out/label-measure/`).
//! `src/label_measure/solution.rs` is a model solution: `cargo test -p labelling --features solution label_measure::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p labelling -- label-measure`.
pub mod run;
