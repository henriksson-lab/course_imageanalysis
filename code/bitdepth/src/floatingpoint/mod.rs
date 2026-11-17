//! Coding exercises for the unit "Floating point".
//!
//! Write your code in `src/floatingpoint/exercise.rs`. Check it with `cargo test -p bitdepth floatingpoint::`, and run it
//! with `cargo run -p bitdepth -- floatingpoint` (pictures in `out/floatingpoint/`).
//! `src/floatingpoint/solution.rs` is a model solution: `cargo test -p bitdepth --features solution floatingpoint::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p bitdepth -- floatingpoint`.
pub mod run;
