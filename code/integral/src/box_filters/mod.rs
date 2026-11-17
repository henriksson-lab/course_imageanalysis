//! Coding exercises for the unit "Box filters & local statistics".
//!
//! Write your code in `src/box_filters/exercise.rs`. Check it with `cargo test -p integral box_filters::`, and run it on
//! the micrograph with `cargo run -p integral -- box-filters` (pictures in `out/box-filters/`).
//! `src/box_filters/solution.rs` is a model solution: `cargo test -p integral --features solution box_filters::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p integral -- box-filters`.
pub mod run;
