//! Coding exercises for the unit "Is it more than chance? Shift curves, randomisation and the ICQ".
//!
//! Write your code in `src/coloc_null/exercise.rs`. Check it with `cargo test -p day44 coloc_null::`, and run it with
//! `cargo run -p day44 -- coloc-null` (results in `out/coloc-null/`).
//! `src/coloc_null/solution.rs` is a model solution: `cargo test -p day44 --features solution coloc_null::`.
//! `src/sim.rs` makes the test data and draws the results; it is not part of the exercise.

pub mod sim;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day44 -- coloc-null`.
pub mod run;
