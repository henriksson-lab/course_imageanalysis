//! Coding exercises for the unit "Point patterns and complete spatial randomness".
//!
//! Write your code in `src/point_patterns/exercise.rs`. Check it with `cargo test -p day19 point_patterns::`, and run it
//! with `cargo run -p day19 -- point-patterns` (results in `out/point-patterns/`).
//! `src/point_patterns/solution.rs` is a model solution: `cargo test -p day19 --features solution point_patterns::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day19 -- point-patterns`.
pub mod run;
