//! Coding exercises for the unit "Principal component analysis".
//!
//! Write your code in `src/pca/exercise.rs`. Check it with `cargo test -p images pca::`, and run it on
//! the micrograph with `cargo run -p images -- pca` (pictures in `out/pca/`).
//! `src/pca/solution.rs` is a model solution: `cargo test -p images --features solution pca::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p images -- pca`.
pub mod run;
