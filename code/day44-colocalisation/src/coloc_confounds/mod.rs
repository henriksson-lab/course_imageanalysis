//! Coding exercises for the unit "What the microscope does to colocalisation".
//!
//! Write your code in `src/coloc_confounds/exercise.rs`. Check it with `cargo test -p day44 coloc_confounds::`, and run it with
//! `cargo run -p day44 -- coloc-confounds` (results in `out/coloc-confounds/`).
//! `src/coloc_confounds/solution.rs` is a model solution: `cargo test -p day44 --features solution coloc_confounds::`.
//! `src/common.rs` makes the test data and draws the results, `src/coloc_confounds/sim.rs` the bead
//! images; they are not part of the exercise.

pub mod sim;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day44 -- coloc-confounds`.
pub mod run;
