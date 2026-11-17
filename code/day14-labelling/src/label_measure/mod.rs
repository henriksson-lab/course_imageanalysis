//! Coding exercises for the unit "Working with label images".
//!
//! Write your code in `src/label_measure/exercise.rs`. Check it with `cargo test -p day14 label_measure::`, and run it on
//! synthetic nuclei with `cargo run -p day14 -- label-measure` (pictures in `out/label-measure/`).
//! `src/label_measure/solution.rs` is a model solution: `cargo test -p day14 --features solution label_measure::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day14 -- label-measure`.
pub mod run;
