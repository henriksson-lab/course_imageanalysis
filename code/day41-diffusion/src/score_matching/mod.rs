//! Coding exercises for the unit "Noise levels, denoisers and the score".
//!
//! Write your code in `src/score_matching/exercise.rs`. Check it with `cargo test -p day41 score_matching::`, and run it
//! with `cargo run -p day41 -- score-matching` (pictures in `out/score-matching/`).
//! `src/score_matching/solution.rs` is a model solution: `cargo test -p day41 --features solution score_matching::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day41 -- score-matching`.
pub mod run;
