//! Coding exercises for the unit "Matching descriptors".
//!
//! Write your code in `src/feature_matching/exercise.rs`. Check it with `cargo test -p day24 feature_matching::`, and run it on
//! the micrograph with `cargo run -p day24 -- feature-matching` (pictures in `out/feature-matching/`).
//! `src/feature_matching/solution.rs` is a model solution: `cargo test -p day24 --features solution feature_matching::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day24 -- feature-matching`.
pub mod run;
