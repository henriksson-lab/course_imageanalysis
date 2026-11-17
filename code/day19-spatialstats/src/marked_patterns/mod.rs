//! Coding exercises for the unit "Two cell types and an uneven density".
//!
//! Write your code in `src/marked_patterns/exercise.rs`. Check it with `cargo test -p day19 marked_patterns::`, and run it
//! with `cargo run -p day19 -- marked-patterns` (results in `out/marked-patterns/`).
//! `src/marked_patterns/solution.rs` is a model solution: `cargo test -p day19 --features solution marked_patterns::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day19 -- marked-patterns`.
pub mod run;
