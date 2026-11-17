//! Coding exercises for the unit "Region properties and moments".
//!
//! Write your code in `src/regionprops/exercise.rs`. Check it with `cargo test -p day14 regionprops::`, and run it on
//! synthetic nuclei and digital discs with `cargo run -p day14 -- regionprops` (pictures in `out/regionprops/`).
//! `src/regionprops/solution.rs` is a model solution: `cargo test -p day14 --features solution regionprops::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day14 -- regionprops`.
pub mod run;
