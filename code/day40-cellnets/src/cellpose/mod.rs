//! Coding exercises for the unit "Flow fields to objects: Cellpose".
//!
//! Write your code in `src/cellpose/exercise.rs`. Check it with `cargo test -p day40 cellpose::`, and run it
//! with `cargo run -p day40 -- cellpose` (pictures in `out/cellpose/`).
//! `src/cellpose/solution.rs` is a model solution: `cargo test -p day40 --features solution cellpose::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day40 -- cellpose`.
pub mod run;
