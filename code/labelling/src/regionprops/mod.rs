//! Coding exercises for the unit "Region properties and moments".
//!
//! Write your code in `src/regionprops/exercise.rs`. Check it with `cargo test -p labelling regionprops::`, and run it on
//! synthetic nuclei and digital discs with `cargo run -p labelling -- regionprops` (pictures in `out/regionprops/`).
//! `src/regionprops/solution.rs` is a model solution: `cargo test -p labelling --features solution regionprops::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p labelling -- regionprops`.
pub mod run;
