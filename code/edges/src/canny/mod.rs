//! Coding exercises for the unit "Edges: Gaussian derivatives and Canny".
//!
//! Write your code in `src/canny/exercise.rs`. Check it with `cargo test -p edges canny::`, and run it on
//! the micrograph with `cargo run -p edges -- canny` (pictures in `out/canny/`).
//! `src/canny/solution.rs` is a model solution: `cargo test -p edges --features solution canny::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p edges -- canny`.
pub mod run;
