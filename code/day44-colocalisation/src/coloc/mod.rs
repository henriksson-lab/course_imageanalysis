//! Coding exercises for the unit "Colocalisation of two channels".
//!
//! Write your code in `src/coloc/exercise.rs`. Check it with `cargo test -p day44 coloc::`, and run it with
//! `cargo run -p day44 -- coloc` (results in `out/coloc/`).
//! `src/coloc/solution.rs` is a model solution: `cargo test -p day44 --features solution coloc::`.
//! `src/common.rs` makes the test data and draws the results; it is not part of the exercise.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day44 -- coloc`.
pub mod run;
