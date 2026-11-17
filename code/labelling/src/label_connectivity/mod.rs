//! Coding exercises for the unit "Connectivity and labelling".
//!
//! Write your code in `src/label_connectivity/exercise.rs`. Check it with `cargo test -p labelling label_connectivity::`, and run it on
//! synthetic nuclei and the micrograph with `cargo run -p labelling -- label-connectivity` (pictures in `out/label-connectivity/`).
//! `src/label_connectivity/solution.rs` is a model solution: `cargo test -p labelling --features solution label_connectivity::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p labelling -- label-connectivity`.
pub mod run;
